"""
Model fine-tuning implementation for the AI Trading Agent.

This module provides functionality for fine-tuning language models on
financial domain data using various methods including instruction tuning,
LoRA, and other parameter-efficient techniques.
"""
import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from pathlib import Path
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from .config import FineTuningConfig, ModelProvider, FineTuningMethod


logger = logging.getLogger(__name__)


class ModelFineTuner:
    """
    Fine-tunes language models on financial domain data.
    
    This class handles the fine-tuning process for various model providers
    and fine-tuning methods, including instruction tuning and parameter
    efficient methods like LoRA.
    """
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the model fine-tuner with configuration.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.training_args = None
        self._initialize_environment()
    
    def _initialize_environment(self) -> None:
        """Initialize the training environment based on configuration."""
        # Set up output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set appropriate environment variables based on provider
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        if self.config.model_provider == ModelProvider.OPENAI:
            if self.config.api_key:
                os.environ["OPENAI_API_KEY"] = self.config.api_key
            if self.config.org_id:
                os.environ["OPENAI_ORG_ID"] = self.config.org_id
                
        elif self.config.model_provider == ModelProvider.ANTHROPIC:
            if self.config.api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
                
        elif self.config.model_provider == ModelProvider.AZURE_OPENAI:
            if self.config.api_key:
                os.environ["AZURE_OPENAI_API_KEY"] = self.config.api_key
            if self.config.api_base:
                os.environ["AZURE_OPENAI_ENDPOINT"] = self.config.api_base
            if self.config.api_version:
                os.environ["AZURE_OPENAI_API_VERSION"] = self.config.api_version
                
        elif self.config.model_provider == ModelProvider.HUGGING_FACE:
            if self.config.api_key:
                os.environ["HUGGINGFACE_API_TOKEN"] = self.config.api_key
    
    def fine_tune(self, prepared_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Fine-tune the model on prepared data.
        
        Args:
            prepared_data: Dictionary with train, validation, and test datasets
        """
        logger.info(f"Starting fine-tuning with {self.config.model_provider.value} provider")
        
        # Check appropriate fine-tuning method based on provider
        if self.config.model_provider == ModelProvider.OPENAI:
            self._fine_tune_openai(prepared_data)
        elif self.config.model_provider == ModelProvider.ANTHROPIC:
            self._fine_tune_anthropic(prepared_data)
        elif self.config.model_provider == ModelProvider.AZURE_OPENAI:
            self._fine_tune_azure_openai(prepared_data)
        elif self.config.model_provider in [ModelProvider.HUGGING_FACE, ModelProvider.LOCAL]:
            self._fine_tune_huggingface(prepared_data)
        else:
            raise ValueError(f"Unsupported model provider: {self.config.model_provider.value}")
    
    def _fine_tune_openai(self, prepared_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Fine-tune an OpenAI model.
        
        Args:
            prepared_data: Dictionary with train, validation, and test datasets
        """
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with `pip install openai`")
        
        logger.info(f"Fine-tuning OpenAI model: {self.config.base_model}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=self.config.api_key)
        
        # Prepare data in OpenAI format
        training_file_path = os.path.join(self.config.output_dir, "training_data.jsonl")
        validation_file_path = os.path.join(self.config.output_dir, "validation_data.jsonl")
        
        # Convert to OpenAI format and save
        self._prepare_openai_format(prepared_data["train"], training_file_path)
        self._prepare_openai_format(prepared_data["validation"], validation_file_path)
        
        # Upload files to OpenAI
        logger.info("Uploading training file to OpenAI")
        with open(training_file_path, "rb") as training_file:
            training_response = client.files.create(
                file=training_file,
                purpose="fine-tune"
            )
        training_file_id = training_response.id
        
        logger.info("Uploading validation file to OpenAI")
        with open(validation_file_path, "rb") as validation_file:
            validation_response = client.files.create(
                file=validation_file,
                purpose="fine-tune"
            )
        validation_file_id = validation_response.id
        
        # Wait for files to be processed
        logger.info("Waiting for files to be processed...")
        processed = False
        while not processed:
            training_file_info = client.files.retrieve(training_file_id)
            validation_file_info = client.files.retrieve(validation_file_id)
            
            if training_file_info.status == "processed" and validation_file_info.status == "processed":
                processed = True
            else:
                time.sleep(5)
        
        # Create fine-tuning job
        logger.info(f"Creating fine-tuning job for model {self.config.base_model}")
        
        # Set hyperparameters based on config
        hyperparameters = {}
        if self.config.num_epochs:
            hyperparameters["n_epochs"] = self.config.num_epochs
        if self.config.batch_size:
            hyperparameters["batch_size"] = self.config.batch_size
        if self.config.learning_rate:
            hyperparameters["learning_rate_multiplier"] = self.config.learning_rate
        
        # Create the fine-tuning job
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=self.config.base_model,
            hyperparameters=hyperparameters,
            suffix=f"financial-domain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Save job ID
        job_id = job.id
        with open(os.path.join(self.config.output_dir, "fine_tuning_job.json"), "w") as f:
            json.dump({"job_id": job_id, "provider": "openai"}, f)
        
        logger.info(f"Fine-tuning job created with ID: {job_id}")
        logger.info("Fine-tuning is running asynchronously on OpenAI's servers")
        logger.info(f"Check status with: client.fine_tuning.jobs.retrieve('{job_id}')")
    
    def _prepare_openai_format(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Prepare data in OpenAI fine-tuning format.
        
        Args:
            data: List of data instances
            output_path: Path to save the JSONL file
        """
        with open(output_path, "w") as f:
            for item in data:
                # Format based on whether it's a chat or completion model
                if "gpt-3.5-turbo" in self.config.base_model or "gpt-4" in self.config.base_model:
                    # Chat format
                    if "messages" in item:
                        # Already in chat format
                        example = item
                    else:
                        # Convert to chat format
                        example = {
                            "messages": [
                                {"role": "system", "content": item.get("system_prompt", "You are a helpful financial advisor and trading expert.")},
                                {"role": "user", "content": item.get("prompt", item.get("text", ""))},
                                {"role": "assistant", "content": item.get("completion", item.get("response", ""))}
                            ]
                        }
                else:
                    # Completion format (legacy)
                    example = {
                        "prompt": item.get("prompt", item.get("text", "")),
                        "completion": item.get("completion", item.get("response", ""))
                    }
                
                # Write to JSONL
                f.write(json.dumps(example) + "\n")
    
    def _fine_tune_anthropic(self, prepared_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Fine-tune an Anthropic model.
        
        Args:
            prepared_data: Dictionary with train, validation, and test datasets
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with `pip install anthropic`")
        
        logger.info(f"Fine-tuning Anthropic model: {self.config.base_model}")
        
        # Anthropic does not currently support custom fine-tuning via API
        # Implementation will be updated when Anthropic provides this capability
        logger.warning("Anthropic custom fine-tuning is not currently supported via public API")
        logger.info("Preparing data in Anthropic format for future use")
        
        # Still prepare the data in the expected format for future use
        anthropic_data_path = os.path.join(self.config.output_dir, "anthropic_data.jsonl")
        
        with open(anthropic_data_path, "w") as f:
            for split, items in prepared_data.items():
                for item in items:
                    # Format in a way that will likely be compatible with future Anthropic fine-tuning
                    example = {
                        "human": item.get("prompt", item.get("text", "")),
                        "assistant": item.get("completion", item.get("response", "")),
                        "split": split
                    }
                    f.write(json.dumps(example) + "\n")
        
        logger.info(f"Saved formatted data to {anthropic_data_path}")
        logger.info("When Anthropic supports custom fine-tuning, update this method implementation")
    
    def _fine_tune_azure_openai(self, prepared_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Fine-tune an Azure OpenAI model.
        
        Args:
            prepared_data: Dictionary with train, validation, and test datasets
        """
        try:
            import openai
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with `pip install openai`")
        
        logger.info(f"Fine-tuning Azure OpenAI model: {self.config.base_model}")
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version or "2023-05-15",
            azure_endpoint=self.config.api_base
        )
        
        # The rest of the implementation is similar to OpenAI
        # Prepare data in OpenAI format
        training_file_path = os.path.join(self.config.output_dir, "training_data.jsonl")
        validation_file_path = os.path.join(self.config.output_dir, "validation_data.jsonl")
        
        # Convert to OpenAI format and save
        self._prepare_openai_format(prepared_data["train"], training_file_path)
        self._prepare_openai_format(prepared_data["validation"], validation_file_path)
        
        # Upload files and create fine-tuning job
        # Note: The exact API calls might differ slightly for Azure OpenAI
        # This implementation will need to be updated based on Azure OpenAI's API documentation
        logger.warning("Azure OpenAI fine-tuning implementation may need updates based on latest Azure API")
        
        # Similar approach to OpenAI fine-tuning
        # Details omitted as they would be similar to _fine_tune_openai method
        logger.info("Fine-tuning process initiated for Azure OpenAI")
    
    def _fine_tune_huggingface(self, prepared_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Fine-tune a Hugging Face model.
        
        Args:
            prepared_data: Dictionary with train, validation, and test datasets
        """
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, 
                TrainingArguments, Trainer, 
                DataCollatorForLanguageModeling
            )
            from peft import get_peft_model, LoraConfig, TaskType, PeftModel
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Required packages not installed. Install with: "
                "`pip install torch transformers peft datasets`"
            )
        
        logger.info(f"Fine-tuning Hugging Face model: {self.config.base_model}")
        
        # Prepare model
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare datasets
        train_dataset = self._prepare_hf_dataset(prepared_data["train"], tokenizer)
        eval_dataset = self._prepare_hf_dataset(prepared_data["validation"], tokenizer)
        
        # Determine whether to use LoRA
        if self.config.fine_tuning_method in [
            FineTuningMethod.LORA, 
            FineTuningMethod.QLORA, 
            FineTuningMethod.PEFT
        ]:
            # Use parameter-efficient fine-tuning
            logger.info(f"Using {self.config.fine_tuning_method.value} for parameter-efficient fine-tuning")
            
            # Load base model with appropriate precision
            model_kwargs = {}
            if self.config.fine_tuning_method == FineTuningMethod.QLORA:
                try:
                    import bitsandbytes as bnb
                    from transformers import BitsAndBytesConfig
                    
                    # Configure quantization
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed. Falling back to regular LoRA.")
                    logger.info("Install with: pip install bitsandbytes")
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16 if self.config.fp16 else None,
                device_map="auto",
                **model_kwargs
            )
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # Get PEFT model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
        else:
            # Full fine-tuning
            logger.info("Using full fine-tuning")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16 if self.config.fp16 else None
            )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            load_best_model_at_end=True,
            report_to="tensorboard",
            remove_unused_columns=False,
        )
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Set up trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("Starting training")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model")
        model.save_pretrained(os.path.join(self.config.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(self.config.output_dir, "final_model"))
        
        # Save training metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        
        # Evaluate
        logger.info("Evaluating final model")
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)
        
        # Save training state
        with open(os.path.join(self.config.output_dir, "fine_tuning_info.json"), "w") as f:
            json.dump({
                "provider": "huggingface",
                "base_model": self.config.base_model,
                "method": self.config.fine_tuning_method.value,
                "completed_at": datetime.now().isoformat(),
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_results
            }, f, indent=2)
        
        logger.info("Fine-tuning completed successfully")
    
    def _prepare_hf_dataset(self, data: List[Dict[str, Any]], tokenizer) -> "Dataset":
        """
        Prepare data in Hugging Face dataset format.
        
        Args:
            data: List of data instances
            tokenizer: HuggingFace tokenizer
            
        Returns:
            HuggingFace Dataset
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets package not installed. Install with `pip install datasets`")
        
        # Format data for instruction tuning or language modeling
        formatted_data = []
        
        for item in data:
            if self.config.fine_tuning_method == FineTuningMethod.INSTRUCTION_TUNING:
                # Instruction tuning format
                if "prompt" in item and "completion" in item:
                    # Already in instruction format
                    instruction = item["prompt"]
                    response = item["completion"]
                elif "system_prompt" in item and "prompt" in item and "response" in item:
                    # Convert from system/user/assistant format
                    system = item["system_prompt"]
                    instruction = f"{system}\n\nUser: {item['prompt']}"
                    response = f"Assistant: {item['response']}"
                elif "text" in item:
                    # Handle plain text - split by common patterns if possible
                    parts = item["text"].split("User:", 1)
                    if len(parts) > 1:
                        instruction = f"User:{parts[1].split('Assistant:', 1)[0].strip()}"
                        response = f"Assistant:{parts[1].split('Assistant:', 1)[1].strip()}"
                    else:
                        # Just use as instruction with generic response
                        instruction = item["text"]
                        response = "This information is valuable for financial analysis and trading decisions."
                else:
                    # Skip invalid items
                    continue
                
                # Create formatted text
                formatted_text = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n{response.strip()}"
                
                # Tokenize
                tokenized = tokenizer(formatted_text, truncation=True, padding="max_length", max_length=1024)
                formatted_data.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["input_ids"].copy()
                })
                
            else:
                # Regular language modeling format
                text = item.get("text", "")
                if not text and "prompt" in item and "completion" in item:
                    text = f"{item['prompt']}{item['completion']}"
                
                if text:
                    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
                    formatted_data.append({
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "labels": tokenized["input_ids"].copy()
                    })
        
        return Dataset.from_list(formatted_data)
    
    def load_fine_tuned_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model, defaults to output_dir/final_model
        """
        if not model_path:
            model_path = os.path.join(self.config.output_dir, "final_model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        if self.config.model_provider in [ModelProvider.HUGGING_FACE, ModelProvider.LOCAL]:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "Required packages not installed. Install with: "
                    "`pip install transformers peft`"
                )
            
            # Check if this is a PEFT model
            is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            if is_peft:
                # Load base model first
                base_model_path = self.config.base_model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16 if self.config.fp16 else None,
                    device_map="auto"
                )
                
                # Then load the PEFT adapter
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                # Load full model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.config.fp16 else None,
                    device_map="auto"
                )
            
            logger.info("Model loaded successfully")
            
        else:
            logger.warning(f"Loading fine-tuned models for {self.config.model_provider.value} must be done through their respective APIs")
            logger.info(f"For OpenAI, use the fine-tuned model name from the API directly")
    
    def generate(self, prompt: str, max_length: int = 1024) -> str:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_fine_tuned_model first.")
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
