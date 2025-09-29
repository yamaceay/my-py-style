from __future__ import annotations
from typing import TYPE_CHECKING, Any
import logging
import sys

if TYPE_CHECKING:
    from config import RuntimeConfig

from core import generate_simulation_data, MyEvaluator, MyCollator, MyTrainer
from contexts import MyClassifierContext, MyDataContext

logger = logging.getLogger(__name__)

def run_training_simulation(args, cfg: "RuntimeConfig") -> dict[str, Any]:
    logger.info("starting_training_simulation", extra={
        "model_name": args.model_name,
        "epochs": cfg.epochs,
        "data_size": args.data_size
    })
    
    train_data, val_data = generate_simulation_data(args.data_size)
    
    labels = ["positive", "negative", "neutral"]
    
    with MyDataContext(train_data, val_data) as data_ctx:
        with MyClassifierContext(args.model_name, labels, cfg) as classifier:
            collator = MyCollator(classifier.tokenizer, max_length=cfg.max_length)
            evaluator = MyEvaluator(labels)
            trainer = MyTrainer(classifier=classifier, config=cfg)
            
            training_results = trainer.train(
                train_data=data_ctx.get_train_data(),
                val_data=data_ctx.get_val_data(),
                collator=collator,
                evaluator=evaluator,
                epochs=cfg.epochs
            )
            
            results = {
                "training": training_results,
                "configuration": {
                    "model_name": args.model_name,
                    "labels": labels,
                    "data_size": args.data_size,
                    "device": cfg.device
                }
            }
            
            logger.info("training_simulation_complete", extra={
                "final_accuracy": training_results["final_accuracy"],
                "best_accuracy": training_results["best_accuracy"]
            })
            
            return results

if __name__ == "__main__":
    from cli import build_parser, configure_logging, parse_args_to_config, validate_args
    
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    validate_args(args)
    cfg = parse_args_to_config(args)     
    
    try:
        logger.info("simulation_starting", extra={
            "command": args.cmd,
            "log_level": cfg.log_level,
            "device": cfg.device
        })
        
        if args.cmd == "train":
            results = run_training_simulation(args, cfg)
        else:
            raise ValueError(f"Command '{args.cmd}' not supported")
        
        logger.info("simulation_complete", extra={"success": True})
        print("Simulation completed successfully!")
        print(f"Final accuracy: {results['training']['final_accuracy']:.3f}")
        
    except Exception as e:
        logger.error("simulation_failed", extra={"error": str(e)})
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)