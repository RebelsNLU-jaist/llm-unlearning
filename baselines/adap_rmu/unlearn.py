import os
import datetime

import numpy as np
import torch
from transformers import AdamW
import tqdm as tqdm
import wandb
from baselines.adap_rmu.utils import load_model, get_params, forward_with_cache, get_data


def run_adaptive_rmu(updated_model, frozen_model, tokenizer, forget_data_list,retain_data_list, args):
    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = args.max_num_batches

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    forget_bio, forget_cyber, retain_bio, retain_cyber = {}, {}, {}, {}
    
    for epoch in range(1):
        with tqdm.tqdm(total=num_batches) as pbar:
            coeffs = {"0": 1.0, "1": 1.0}
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)

                control_vec = control_vectors_list[topic_idx]
                
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                )
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                
                if idx == 0:
                    coeffs["0"] = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item() * args.scale
                elif idx == 1:
                    coeffs["1"] = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item() * args.scale
                else:
                    pass
                print(coeffs)
                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec * coeffs[f"{topic_idx}"]
                )

                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= args.alpha[topic_idx]

                # Update model
                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                
                # ======= Logging ======
                if args.verbose:
                    frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                    unlearn_cosine= torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                    
                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(f"Topic {topic_idx} updated_forget_activations.norm=",torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_forget_activations.norm=",torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} updated_retain_activations.norm=",torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    print(f"Topic {topic_idx} frozen_retain_activations.norm=",torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                    if topic_idx == 0:
                        forget_bio["updated_forget"] = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                        forget_bio["frozen_forget"] = torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                        retain_bio["updated_retain"] = torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                        retain_bio["frozen_retain"] = torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                    elif topic_idx == 1:
                        forget_cyber["updated_forget"] = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                        forget_cyber["frozen_forget"] = torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                        retain_cyber["updated_retain"] = torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                        retain_cyber["frozen_retain"] = torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item()
                    else:
                        print("unsupported")
                wandb.log({"forget_bio": forget_bio, "retain_bio": retain_bio, "forget_cyber": forget_cyber, "retain_cyber": retain_cyber, "loss": loss.item(), "unlearn_loss": unlearn_loss.item(), "retain_loss": retain_loss.item()})
                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        path = f"checkpoints/rmu/adaptive_{args.model_name_or_path}_alpha-{int(args.alpha[0])}-{int(args.alpha[0])}_batches-{num_batches}_layer-{args.layer_id}_scale-{args.scale}"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str, default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--output_dir", type=str, default=None)
    ### Data arguments
    parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext", help="comma-separated list of corpora to retain",)
    parser.add_argument("--forget_corpora", type=str, default="bio-forget-corpus,cyber-forget-corpus", help="comma-separated list of corpora to forget",)
    ### rmu hyperparameters
    parser.add_argument("--alpha", type=str, default="1200,1200", help="retain weight")
    parser.add_argument("--steering_coeffs", type=str, default="1,1", help="Steer vector weight in order of topic")
    parser.add_argument('--scale', type=float, default=5.0)

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=150)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 


if __name__ == "__main__":
    args = get_args()
    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    coeff = args.steering_coeffs
    wandb.init(project="unlearn", name=f"adaptive_{args.model_name_or_path}_unlearn-layer={args.layer_id}_updated-layer={args.layer_ids}_step={args.max_num_batches}_scale-{args.scale}")

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)

    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_adaptive_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )