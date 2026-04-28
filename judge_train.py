import asyncio
import os
import shutil

from dotenv import load_dotenv
from tinker_cookbook import model_info
from tinker_cookbook.supervised import train as sl_train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# This line bypasses the need for the "useEnvFile" setting
load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(SCRIPT_DIR, "judge_prompts_and_answers.jsonl")  # JSONL with "messages" arrays
LOG_DIR       = os.path.join(SCRIPT_DIR, "last_run_metadata")      # checkpoint info written here
MODEL_NAME    = "meta-llama/Llama-3.1-8B-Instruct"                 # base model to fine-tune

NUM_EPOCHS    = 3       # full passes over the training data
BATCH_SIZE    = 4       # examples processed per gradient update
LORA_RANK     = 16      # LoRA adapter rank — higher = more expressive, more memory
LEARNING_RATE = 1e-4

# ── Training ──────────────────────────────────────────────────────────────────

async def main() -> None:
    # Wipe the log dir so checkpoint_utils doesn't pick up metadata from a prior run
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)

    # The renderer controls how chat messages are serialized into tokens for this
    # model family (e.g. Llama 3 uses a specific <|start_header_id|> format)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    # Common tokenization/batching settings shared by the dataset builder and trainer
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=16384,       # truncate sequences longer than this
        batch_size=BATCH_SIZE,
        train_on_what=None,     # None = train on all tokens (both user and assistant turns)
    )

    # Load conversation data from a JSONL file where each line has a "messages" array
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=DATASET_PATH,
        test_size=0,        # no held-out eval split
        shuffle_seed=0,
    )

    cfg = sl_train.Config(
        log_path=LOG_DIR,
        model_name=MODEL_NAME,
        dataset_builder=dataset_builder,
        learning_rate=LEARNING_RATE,
        lr_schedule="linear",   # decay LR linearly to 0 over training
        num_epochs=NUM_EPOCHS,
        lora_rank=LORA_RANK,
        save_every=20,          # save a checkpoint every 20 gradient steps
        eval_every=0,           # 0 = skip eval during training
    )

    await sl_train.main(cfg)

    # After training, read the checkpoint metadata to surface the sampler path.
    # Copy this path into FINE_TUNED_PATH in test.py to run inference.
    from tinker_cookbook import checkpoint_utils
    info = checkpoint_utils.get_last_checkpoint(LOG_DIR)
    if info:
        print("\n=== Training complete ===")
        print(f"Sampler path: {info.sampler_path}")
        print("\nPaste this path into FINE_TUNED_PATH in test.py.")


if __name__ == "__main__":
    asyncio.run(main())
