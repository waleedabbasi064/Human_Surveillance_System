from args import create_exp_dirs
from args import init_parser, init_sub_args, load_config, merge_config
import torch
import random
import numpy as np
import copy
import csv

from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params, Trainer, init_optimizer, init_scheduler, CostumLoss
from utils.data_utils import trans_list
from utils.eval import score_dataset, combined_score_dataset
import yaml
import os
from models import *
from utils.tokenizer import Tokenizer
from utils.eval import eval
import subprocess


def _ensure_remote_file(path: str | None) -> str | None:
    """If `path` does not exist locally, attempt to download it.

    - If `HF_WEIGHTS_REPO` env is set, try to download the basename from that repo.
    - Else if `WEIGHTS_BASE_URL` is set, attempt HTTP GET.
    Returns a path that exists locally or None if unresolved.
    """
    if not path:
        return None
    if os.path.exists(path):
        return path

    basename = os.path.basename(path)
    hf_repo = os.environ.get("HF_WEIGHTS_REPO")
    hf_token = os.environ.get("HF_TOKEN")
    print(f"[CHECKPOINT] Local path missing: {path}")
    print(f"[CHECKPOINT] Resolving basename: {basename}")
    print(f"[CHECKPOINT] HF_WEIGHTS_REPO={hf_repo}")
    print(f"[CHECKPOINT] WEIGHTS_BASE_URL={os.environ.get('WEIGHTS_BASE_URL')}")

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download

            kwargs = {"repo_id": hf_repo, "filename": basename, "cache_dir": "/tmp", "force_filename": basename}
            if hf_token:
                kwargs["token"] = hf_token
            target = hf_hub_download(**kwargs)
            print(f"[CHECKPOINT] HF hub download target: {target}")
            if os.path.exists(target):
                return target
            print(f"[CHECKPOINT] Downloaded file not found at target: {target}")
        except Exception as e:
            print(f"Failed to download from HF hub: {e}")

    base_url = os.environ.get("WEIGHTS_BASE_URL")
    if base_url:
        try:
            import requests

            url = base_url.rstrip("/") + "/" + basename
            print(f"[CHECKPOINT] Downloading from URL: {url}")
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                out = os.path.join("/tmp", basename)
                with open(out, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                print(f"[CHECKPOINT] Downloaded file to: {out}")
                return out
            else:
                print(f"HTTP download failed: {resp.status_code} {resp.reason}")
        except Exception as e:
            print(f"Failed to download from WEIGHTS_BASE_URL: {e}")

    print("[CHECKPOINT] Could not resolve remote file; no HF_WEIGHTS_REPO or WEIGHTS_BASE_URL worked.")
    return None


def _build_score_rows(scores, metadata=None, threshold=None):
    rows = []
    if metadata is None:
        metadata = []
    metadata_len = len(metadata)

    for idx, score in enumerate(scores):
        row = {
            "segment": idx,
            "timeline_index": idx,
            "scene_id": "",
            "clip_id": "",
            "person_id": "",
            "start_frame": "",
            "score": float(score),
            "predicted": "",
        }
        if threshold is not None:
            row["predicted"] = int(float(score) > threshold)

        if idx < metadata_len:
            try:
                meta = list(metadata[idx])
                if len(meta) >= 4:
                    scene_id, clip_id, person_id, start_frame = meta[:4]
                    row["scene_id"] = scene_id
                    row["clip_id"] = clip_id
                    row["person_id"] = person_id
                    row["start_frame"] = int(start_frame)
                    row["timeline_index"] = int(start_frame)
            except Exception:
                pass

        rows.append(row)
    return rows


def main ():
    parser = init_parser()
    args = parser.parse_args()
    cfg = load_config(getattr(args, "config", None))
    args = merge_config(args, cfg, parser)

    # Provide a safe default pointing to your HF model repo if the Space did not set it
    os.environ.setdefault("HF_WEIGHTS_REPO", "shahzaib7788/pose-weights")

    def save_raw_scores(scores, label_suffix="", threshold=None, metadata=None):
        save_dir = args.save_results_dir or os.path.join(args.model_save_dir, "evaluation_results")
        os.makedirs(save_dir, exist_ok=True)
        label = label_suffix or args.branch.lower()
        out_path = os.path.join(save_dir, f"{label}_scores.csv")
        rows = _build_score_rows(scores, metadata=metadata, threshold=threshold)
        fieldnames = ["segment", "timeline_index", "scene_id", "clip_id", "person_id", "start_frame", "score", "predicted"]
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Scores saved to {out_path}")
        return out_path

    def save_score_plot(scores, label_suffix="", threshold=None, metadata=None):
        if not getattr(args, "plot_results", False):
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = args.plot_results_dir or args.save_results_dir or os.path.join(args.model_save_dir, "evaluation_plots")
        os.makedirs(plot_dir, exist_ok=True)
        label = label_suffix or args.branch.lower()
        out_path = os.path.join(plot_dir, f"{label}_scores_plot.png")
        rows = _build_score_rows(scores, metadata=metadata, threshold=threshold)

        if not rows:
            print("No rows available to plot.")
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        person_groups = {}
        for row in rows:
            person_key = row["person_id"] if row["person_id"] != "" else "all"
            person_groups.setdefault(person_key, []).append(row)

        for person_key, group_rows in person_groups.items():
            group_rows = sorted(group_rows, key=lambda item: item["timeline_index"])
            x_vals = [item["timeline_index"] for item in group_rows]
            y_vals = [item["score"] for item in group_rows]
            label_name = f"person_{person_key}" if person_key != "all" else label
            ax.plot(x_vals, y_vals, label=label_name, alpha=0.8)

        if threshold is not None:
            ax.axhline(float(threshold), color="red", linestyle="--", linewidth=1.2, label=f"threshold={float(threshold):.4f}")

        ax.set_title(f"{label.upper()} score / loss over video timeline")
        ax.set_xlabel("Timeline (start frame)")
        ax.set_ylabel("Score / Loss")
        ax.grid(True, linestyle="--", alpha=0.35)
        if len(person_groups) <= 12:
            ax.legend(loc="upper right", fontsize="small", ncol=2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {out_path}")
        return out_path

    # Fallback to CPU if CUDA is unavailable or fails to initialize
    if str(args.device).startswith('cuda'):
        try:
            _ = torch.cuda.device_count()
        except Exception as e:
            print(f"CUDA init failed ({e}), falling back to cpu")
            args.device = 'cpu'
        else:
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to cpu")
                args.device = 'cpu'

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    # Normalize friendly branch labels to internal branch names
    branch_map = {
        "Reconstruction Model": "SPARTA_C",
        "Future trajecory prediction model": "SPARTA_F",
        "Hybrid": "SPARTA_H",
    }
    args.branch = branch_map.get(args.branch, args.branch)

    pretrained_model = (args.mode == "test") or (args.branch == "SPARTA_H") or bool(vars(args).get('model_ckpt_dir'))
    recon_encoder = vars(args).get('recon_encoder_path', None)

    if args.mode == "test":
        if args.branch == "SPARTA_H":
            if not args.model_ckpt_C or not args.model_ckpt_F:
                raise ValueError("Test mode with SPARTA_H requires --model_ckpt_C and --model_ckpt_F.")
        else:
            if not args.model_ckpt_dir:
                raise ValueError("Test mode requires --model_ckpt_dir for the selected branch.")

    if args.branch == "SPARTA_H":
        args_c = copy.deepcopy(args)
        args_c.branch = "SPARTA_C"
        args_f = copy.deepcopy(args)
        args_f.branch = "SPARTA_F"
        dataset_F, loader_F = get_dataset_and_loader(args_f, trans_list=trans_list, only_test=pretrained_model)
        dataset, loader = get_dataset_and_loader(args_c, trans_list=trans_list, only_test=pretrained_model)
    else:
        dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=pretrained_model)

    expand_ratio = 1
    extra_dim = 0
    if args.relative:
        expand_ratio = 2

    if model_args.token_config == "kps":
        input_dim = model_args.seg_len*2
    elif model_args.token_config == "2ds":
        input_dim = model_args.seg_len
    elif model_args.token_config == "st":
        input_dim = 2
    else:
        if args.traj:
            if args.relative:
                extra_dim = 4
            else:
                extra_dim = 2
        input_dim = args.num_kp*2

    target_loader = loader.get('train') or loader.get('test')
    if target_loader:
        print(f"[INFO] Dataset loader is ready: {type(target_loader)}")

    Transformer_model = None
    tokenizer = None
    tokenizer_c = None
    tokenizer_f = None
    if args.branch == "SPARTA_C":
        Transformer_model = SPARTA_C(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device, dropout=args.dropout)
        tokenizer = Tokenizer(args=args)
    elif args.branch == "SPARTA_F":
        Transformer_model = SPARTA_F(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device)
        tokenizer = Tokenizer(args=args)
    elif args.branch == "SPARTA_H":
        Transformer_model = SPARTA_H(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device, dropout=args.dropout)
        tokenizer_f = Tokenizer(args_f)
        tokenizer_c = Tokenizer(args_c)
    else:
        raise ValueError(f"Unsupported branch '{args.branch}'. Expected SPARTA_C, SPARTA_F, or SPARTA_H.")

    if not pretrained_model:
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        if recon_encoder is not None:
            checkpoint = torch.load(args.recon_encoder_path)
            model_dict = Transformer_model.state_dict()
            encoder_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if 'encoder' in key:
                    encoder_state_dict[key] = value

            for name, param in encoder_state_dict.items():
                if name not in model_dict:
                    continue

                param = param.data
                model_dict[name].copy_(param)

            encoder_layers = Transformer_model.encoder.layers
            for l in encoder_layers:
                l.trainable = False
            print("Frozen Reconstruction Encoder Loaded!")

        arguments = vars(args)
        with open(args.model_save_dir + '/' + 'arguments.yaml', 'w') as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(args.sched, lr=args.model_lr, epochs=args.epochs)
        trainer = Trainer(model_args, Transformer_model, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                                scheduler_f=ae_scheduler_f)
        trainer.train(checkpoint_filename='trans', args=args)

        if args.skip_final_eval:
            print("Final evaluation skipped (--skip_final_eval).")
            return
        if loader['test'] is None or len(loader['test'].dataset) == 0:
            print("No test data available; skipping final evaluation.")
            return

    else:
        # Ensure checkpoints are available locally (download from HF hub or HTTP if configured)
        if args.branch == "SPARTA_H":
            args.model_ckpt_C = _ensure_remote_file(args.model_ckpt_C) or args.model_ckpt_C
            args.model_ckpt_F = _ensure_remote_file(args.model_ckpt_F) or args.model_ckpt_F
            sparta_c_weights = torch.load(args.model_ckpt_C, map_location=args.device)
            sparta_f_weights = torch.load(args.model_ckpt_F, map_location=args.device)
            Transformer_model.CTD.load_state_dict(sparta_c_weights['state_dict'])
            Transformer_model.FTD.load_state_dict(sparta_f_weights['state_dict'])
        else:
            args.model_ckpt_dir = _ensure_remote_file(args.model_ckpt_dir) or args.model_ckpt_dir
            checkpoint = torch.load(args.model_ckpt_dir,  map_location=args.device)
            Transformer_model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully!')
        Transformer_model.to(args.device)
        loss_func = CostumLoss(model_args.loss, a=model_args.a, b=model_args.b, c=model_args.c, d=model_args.d)

        if args.branch == "SPARTA_H":
            print ("*********************SPARTA_C************************")
            eval_loss = eval (args_c, model_args, Transformer_model.CTD, tokenizer_c, loss_func, loader)
            if args.no_metrics:
                save_raw_scores(eval_loss, "sparta_c", threshold=args.eer_threshold_c, metadata=dataset['test'].metadata)
                save_score_plot(eval_loss, "sparta_c", threshold=args.eer_threshold_c, metadata=dataset['test'].metadata)
            else:
                auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), dataset['test'].metadata, args=args_c)
                print('AUC ROC: {}'.format(auc_roc))
                print('AUC PR: {}'.format(auc_pr))
                print('EER: {}'.format(eer))
                print('EER TH: {}'.format(eer_th))
                print('10ER: {}'.format(fpr_at_target_fnr))
                print('10ER TH: {}'.format(threshold_at_target_fnr))
                if args.plot_results:
                    save_raw_scores(eval_loss, "sparta_c", threshold=eer_th, metadata=dataset['test'].metadata)
                    save_score_plot(eval_loss, "sparta_c", threshold=eer_th, metadata=dataset['test'].metadata)

            print ("*********************SPARTA_F************************")
            args.branch = "SPARTA_F"
            eval_loss_ = eval (args_f, model_args, Transformer_model.FTD, tokenizer_f, loss_func, loader_F)
            if args.no_metrics:
                save_raw_scores(eval_loss_, "sparta_f", threshold=args.eer_threshold_f, metadata=dataset_F['test'].metadata)
                save_score_plot(eval_loss_, "sparta_f", threshold=args.eer_threshold_f, metadata=dataset_F['test'].metadata)
            else:
                auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss_), dataset_F['test'].metadata, args=args_f)
                print('AUC ROC: {}'.format(auc_roc))
                print('AUC PR: {}'.format(auc_pr))
                print('EER: {}'.format(eer))
                print('EER TH: {}'.format(eer_th))
                print('10ER: {}'.format(fpr_at_target_fnr))
                print('10ER TH: {}'.format(threshold_at_target_fnr))
                if args.plot_results:
                    save_raw_scores(eval_loss_, "sparta_f", threshold=eer_th, metadata=dataset_F['test'].metadata)
                    save_score_plot(eval_loss_, "sparta_f", threshold=eer_th, metadata=dataset_F['test'].metadata)

            if args.no_metrics:
                return
            print ("*********************SPARTA_H************************")
            args.branch = "SPARTA_H"
            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = combined_score_dataset(np.array(eval_loss), np.array(eval_loss_), dataset['test'].metadata, dataset_F['test'].metadata, args=args)
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
        else:
            eval_loss = eval (args, model_args, Transformer_model, tokenizer, loss_func,  loader)
            if args.no_metrics:
                threshold = args.eer_threshold_c if args.branch == "SPARTA_C" else args.eer_threshold_f
                save_raw_scores(eval_loss, args.branch.lower(), threshold=threshold, metadata=dataset['test'].metadata)
                save_score_plot(eval_loss, args.branch.lower(), threshold=threshold, metadata=dataset['test'].metadata)
                return
            auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), dataset['test'].metadata, args=args)
            print('AUC ROC: {}'.format(auc_roc))
            print('AUC PR: {}'.format(auc_pr))
            print('EER: {}'.format(eer))
            print('EER TH: {}'.format(eer_th))
            print('10ER: {}'.format(fpr_at_target_fnr))
            print('10ER TH: {}'.format(threshold_at_target_fnr))
            if args.plot_results:
                save_raw_scores(eval_loss, args.branch.lower(), threshold=eer_th, metadata=dataset['test'].metadata)
                save_score_plot(eval_loss, args.branch.lower(), threshold=eer_th, metadata=dataset['test'].metadata)


if __name__ == '__main__':
    main()
