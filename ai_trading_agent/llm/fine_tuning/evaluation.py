"""
Model evaluation tools for financial LLM fine-tuning.

This module provides tools for evaluating the performance of fine-tuned
language models on financial domain tasks, including specialized metrics
for financial accuracy and regulatory compliance.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import FineTuningConfig, ModelProvider


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates fine-tuned models on financial domain-specific tasks.
    
    This class provides methods for assessing model performance on
    financial domain tasks, including accuracy, domain knowledge,
    and regulatory compliance.
    """
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the model evaluator with configuration.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.evaluation_results = {}
        self.metrics = {}
    
    def evaluate(self, test_data: List[Dict[str, Any]], model_provider: str, model_id: str) -> Dict[str, float]:
        """
        Evaluate a fine-tuned model on test data.
        
        Args:
            test_data: List of test data examples
            model_provider: Model provider (e.g., "openai", "huggingface")
            model_id: Model ID or path
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model {model_id} from provider {model_provider}")
        
        # Initialize metrics
        self.metrics = {
            "financial_accuracy": 0.0,
            "regulatory_compliance": 0.0,
            "reasoning_quality": 0.0,
            "factual_consistency": 0.0,
            "domain_specificity": 0.0,
            "overall_score": 0.0
        }
        
        # Create client based on provider
        client = self._get_client(model_provider, model_id)
        
        # Evaluate on test data
        results = []
        for item in tqdm(test_data, desc="Evaluating"):
            try:
                result = self._evaluate_example(item, client, model_provider)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(results)
        self.metrics.update(metrics)
        
        # Calculate overall score
        self.metrics["overall_score"] = self._calculate_overall_score(self.metrics)
        
        # Save evaluation results
        self.evaluation_results = {
            "model_provider": model_provider,
            "model_id": model_id,
            "metrics": self.metrics,
            "example_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file
        evaluation_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        with open(evaluation_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {evaluation_path}")
        logger.info(f"Overall score: {self.metrics['overall_score']:.4f}")
        
        return self.metrics
    
    def _get_client(self, model_provider: str, model_id: str) -> Any:
        """
        Get the appropriate client for the model provider.
        
        Args:
            model_provider: Model provider name
            model_id: Model ID or path
            
        Returns:
            Client object for making inference requests
        """
        if model_provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with `pip install openai`")
                
        elif model_provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Install with `pip install anthropic`")
                
        elif model_provider == "azure_openai":
            try:
                from openai import AzureOpenAI
                return AzureOpenAI(
                    api_key=self.config.api_key,
                    api_version=self.config.api_version or "2023-05-15",
                    azure_endpoint=self.config.api_base
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with `pip install openai`")
                
        elif model_provider in ["huggingface", "local"]:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                
                # Check if this is a PEFT model
                is_peft = os.path.exists(os.path.join(model_id, "adapter_config.json"))
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
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
                    model = PeftModel.from_pretrained(base_model, model_id)
                else:
                    # Load full model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if self.config.fp16 else None,
                        device_map="auto"
                    )
                
                return {"model": model, "tokenizer": tokenizer}
            except ImportError:
                raise ImportError(
                    "Required packages not installed. Install with: "
                    "`pip install torch transformers peft`"
                )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
    
    def _evaluate_example(self, example: Dict[str, Any], client: Any, model_provider: str) -> Dict[str, Any]:
        """
        Evaluate a single example.
        
        Args:
            example: Test example
            client: Client for making inference requests
            model_provider: Name of the model provider
            
        Returns:
            Evaluation result for the example
        """
        # Get input prompt
        prompt = self._prepare_prompt(example, model_provider)
        
        # Get model response
        response = self._get_model_response(prompt, client, model_provider)
        
        # Get reference answer
        reference = example.get("completion", example.get("response", ""))
        
        # Evaluate response
        evaluation = {
            "prompt": prompt,
            "response": response,
            "reference": reference,
            "metrics": {}
        }
        
        # Calculate metrics
        evaluation["metrics"]["financial_accuracy"] = self._evaluate_financial_accuracy(response, reference, example)
        evaluation["metrics"]["regulatory_compliance"] = self._evaluate_regulatory_compliance(response, example)
        evaluation["metrics"]["reasoning_quality"] = self._evaluate_reasoning_quality(response, reference)
        evaluation["metrics"]["factual_consistency"] = self._evaluate_factual_consistency(response, reference)
        evaluation["metrics"]["domain_specificity"] = self._evaluate_domain_specificity(response)
        
        # Calculate overall score for this example
        evaluation["metrics"]["overall_score"] = self._calculate_overall_score(evaluation["metrics"])
        
        return evaluation
    
    def _prepare_prompt(self, example: Dict[str, Any], model_provider: str) -> str:
        """
        Prepare the prompt for the model.
        
        Args:
            example: Test example
            model_provider: Name of the model provider
            
        Returns:
            Formatted prompt
        """
        if "prompt" in example:
            # Direct prompt
            return example["prompt"]
        elif "text" in example:
            # Handle text-only examples
            return example["text"]
        elif "system_prompt" in example and "prompt" in example:
            # System + user prompt format
            if model_provider == "anthropic":
                return f"{example['system_prompt']}\n\nHuman: {example['prompt']}\n\nAssistant:"
            else:
                return f"{example['system_prompt']}\nUser: {example['prompt']}\nAssistant:"
        else:
            # Default to text field or empty
            return example.get("text", "")
    
    def _get_model_response(self, prompt: str, client: Any, model_provider: str) -> str:
        """
        Get response from the model.
        
        Args:
            prompt: Input prompt
            client: Client for making inference requests
            model_provider: Name of the model provider
            
        Returns:
            Model response
        """
        try:
            if model_provider == "openai":
                response = client.chat.completions.create(
                    model=self.config.base_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                )
                return response.choices[0].message.content
                
            elif model_provider == "anthropic":
                response = client.completions.create(
                    model=self.config.base_model,
                    prompt=prompt,
                    max_tokens_to_sample=1024,
                    temperature=0.7
                )
                return response.completion
                
            elif model_provider == "azure_openai":
                response = client.chat.completions.create(
                    model=self.config.base_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                )
                return response.choices[0].message.content
                
            elif model_provider in ["huggingface", "local"]:
                model = client["model"]
                tokenizer = client["tokenizer"]
                
                # Encode input
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=1024,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7,
                        num_return_sequences=1
                    )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the input prompt from the output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                return generated_text
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")
                
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return f"Error: {str(e)}"
    
    def _evaluate_financial_accuracy(self, response: str, reference: str, example: Dict[str, Any]) -> float:
        """
        Evaluate financial accuracy of the response.
        
        Args:
            response: Model response
            reference: Reference answer
            example: Test example with metadata
            
        Returns:
            Financial accuracy score (0-1)
        """
        # This is a simplified implementation
        # In a production system, this would use a more sophisticated approach
        
        # Check for financial terms in reference that should be in response
        financial_terms = [
            "market", "trade", "stock", "bond", "option", "futures", "forex",
            "hedge", "portfolio", "asset", "liability", "equity", "debt",
            "dividend", "yield", "interest", "volatility", "risk", "return",
            "profit", "loss", "bull", "bear", "trend", "analysis", "valuation",
            "fundamental", "technical", "indicator", "ratio", "balance sheet",
            "income statement", "cash flow", "earnings", "revenue", "expense"
        ]
        
        # Custom financial terms for this example (if provided)
        if "financial_terms" in example:
            if isinstance(example["financial_terms"], list):
                financial_terms.extend(example["financial_terms"])
            elif isinstance(example["financial_terms"], str):
                financial_terms.extend(example["financial_terms"].split(","))
        
        # Count financial terms in reference and response
        ref_terms = sum(1 for term in financial_terms if term.lower() in reference.lower())
        resp_terms = sum(1 for term in financial_terms if term.lower() in response.lower())
        
        if ref_terms == 0:
            # If reference doesn't have financial terms, default to general comparison
            # Simple cosine similarity could be used here
            # For now, assign a default score
            return 0.8
        
        # Calculate coverage ratio
        coverage = min(1.0, resp_terms / ref_terms) if ref_terms > 0 else 0.0
        
        # Check for numerical values in reference that should be in response
        # Extract numbers from both texts
        import re
        ref_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', reference)
        resp_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        
        # Calculate numerical accuracy
        # Simple approach: check how many reference numbers appear in response
        number_match = sum(1 for num in ref_numbers if num in resp_numbers)
        number_accuracy = min(1.0, number_match / len(ref_numbers)) if ref_numbers else 1.0
        
        # Combine metrics (weighted average)
        return 0.7 * coverage + 0.3 * number_accuracy
    
    def _evaluate_regulatory_compliance(self, response: str, example: Dict[str, Any]) -> float:
        """
        Evaluate regulatory compliance of the response.
        
        Args:
            response: Model response
            example: Test example with metadata
            
        Returns:
            Regulatory compliance score (0-1)
        """
        # This is a simplified implementation
        # In a production system, this would use a more sophisticated approach
        
        # List of regulatory disclaimers and compliance phrases
        compliance_phrases = [
            "not financial advice",
            "consult a financial advisor",
            "past performance is not indicative of future results",
            "investment involves risk",
            "this information is for educational purposes only",
            "regulatory",
            "compliance",
            "regulation",
            "regulated",
            "sec",
            "finra",
            "disclosure",
            "disclaimer"
        ]
        
        # Custom compliance requirements for this example (if provided)
        if "compliance_requirements" in example:
            if isinstance(example["compliance_requirements"], list):
                compliance_phrases.extend(example["compliance_requirements"])
            elif isinstance(example["compliance_requirements"], str):
                compliance_phrases.extend(example["compliance_requirements"].split(","))
        
        # Count compliance phrases in the response
        compliance_count = sum(1 for phrase in compliance_phrases if phrase.lower() in response.lower())
        
        # Check for problematic phrases (giving direct financial advice)
        problematic_phrases = [
            "you should buy",
            "you should sell",
            "guaranteed return",
            "guaranteed profit",
            "risk-free investment",
            "will definitely increase",
            "will definitely decrease"
        ]
        
        # Count problematic phrases
        problem_count = sum(1 for phrase in problematic_phrases if phrase.lower() in response.lower())
        
        # Calculate compliance score
        # Simple approach: reward compliance phrases, penalize problematic ones
        base_score = min(1.0, compliance_count / 3)  # Expect at least 3 compliance elements
        penalty = min(1.0, problem_count * 0.25)  # Each problem reduces score by up to 25%
        
        return max(0.0, base_score - penalty)
    
    def _evaluate_reasoning_quality(self, response: str, reference: str) -> float:
        """
        Evaluate the quality of reasoning in the response.
        
        Args:
            response: Model response
            reference: Reference answer
            
        Returns:
            Reasoning quality score (0-1)
        """
        # This is a simplified implementation
        # In a production system, this would use a more sophisticated approach
        
        # Check for reasoning indicators
        reasoning_indicators = [
            "because", "therefore", "thus", "as a result", "consequently",
            "due to", "since", "given that", "this suggests", "this indicates",
            "this implies", "based on", "analysis", "evidence", "data shows",
            "research indicates", "historically", "typically", "generally",
            "consider", "examining", "looking at", "evaluating"
        ]
        
        # Count reasoning indicators
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator.lower() in response.lower())
        
        # Length-normalized score (to avoid favoring very long responses)
        response_words = len(response.split())
        reference_words = len(reference.split())
        
        # Calculate basic reasoning score
        base_score = min(1.0, indicator_count / 5)  # Expect at least 5 reasoning elements
        
        # Adjust based on response length compared to reference
        length_score = 1.0
        if response_words < reference_words * 0.5:
            # Too short (less than half of reference)
            length_score = response_words / (reference_words * 0.5)
        elif response_words > reference_words * 2:
            # Too long (more than twice reference)
            length_score = 1.0 - min(0.5, (response_words - reference_words * 2) / (reference_words * 2))
        
        # Combine scores
        return base_score * length_score
    
    def _evaluate_factual_consistency(self, response: str, reference: str) -> float:
        """
        Evaluate factual consistency of the response compared to reference.
        
        Args:
            response: Model response
            reference: Reference answer
            
        Returns:
            Factual consistency score (0-1)
        """
        # This is a simplified implementation
        # In a production system, this would use a more sophisticated approach like NLI models
        
        # Convert to lowercase for comparison
        response_lower = response.lower()
        reference_lower = reference.lower()
        
        # Split into sentences (crude approximation)
        import re
        ref_sentences = re.split(r'[.!?]+', reference_lower)
        ref_sentences = [s.strip() for s in ref_sentences if s.strip()]
        
        # Check for consistency
        # For each key fact (sentence) in reference, check if it's contained or contradicted
        consistency_scores = []
        
        for sentence in ref_sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
                
            # Check for key phrases from this sentence
            words = sentence.split()
            key_phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            # Find matches
            matches = sum(1 for phrase in key_phrases if phrase in response_lower)
            
            # Calculate sentence consistency
            if matches > 0:
                consistency = min(1.0, matches / max(1, len(key_phrases) / 3))
                consistency_scores.append(consistency)
            else:
                consistency_scores.append(0.0)
        
        # Calculate overall consistency
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 0.5  # Default if no meaningful comparison possible
    
    def _evaluate_domain_specificity(self, response: str) -> float:
        """
        Evaluate domain specificity of the response for financial content.
        
        Args:
            response: Model response
            
        Returns:
            Domain specificity score (0-1)
        """
        # This is a simplified implementation
        
        # Check for domain-specific terminology
        financial_jargon = [
            "alpha", "beta", "sharpe ratio", "sortino ratio", "drawdown",
            "volatility", "standard deviation", "correlation", "covariance",
            "portfolio optimization", "efficient frontier", "capital asset pricing model",
            "arbitrage", "derivatives", "options pricing", "black-scholes",
            "monte carlo simulation", "value at risk", "technical analysis",
            "fundamental analysis", "discounted cash flow", "price-to-earnings",
            "earnings per share", "dividend yield", "market capitalization",
            "liquidity", "solvency", "leverage", "debt-to-equity", "return on equity",
            "return on assets", "net present value", "internal rate of return",
            "yield curve", "term structure", "duration", "convexity",
            "market maker", "bid-ask spread", "order book", "limit order",
            "market order", "stop loss", "time-weighted average price",
            "volume-weighted average price", "market microstructure"
        ]
        
        # Count financial jargon terms
        jargon_count = sum(1 for term in financial_jargon if term.lower() in response.lower())
        
        # Calculate domain specificity score
        return min(1.0, jargon_count / 5)  # Expect at least 5 domain-specific terms
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall metrics from individual example results.
        
        Args:
            results: List of evaluation results for examples
            
        Returns:
            Dictionary of average metrics
        """
        # Initialize metrics
        metrics = {
            "financial_accuracy": 0.0,
            "regulatory_compliance": 0.0,
            "reasoning_quality": 0.0,
            "factual_consistency": 0.0,
            "domain_specificity": 0.0
        }
        
        # Calculate averages
        for metric in metrics:
            values = [result["metrics"][metric] for result in results if metric in result["metrics"]]
            metrics[metric] = sum(values) / len(values) if values else 0.0
        
        return metrics
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score from individual metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Overall score (0-1)
        """
        # Weights for each metric
        weights = {
            "financial_accuracy": 0.3,
            "regulatory_compliance": 0.2,
            "reasoning_quality": 0.2,
            "factual_consistency": 0.2,
            "domain_specificity": 0.1
        }
        
        # Calculate weighted average
        score = sum(
            metrics.get(metric, 0.0) * weight 
            for metric, weight in weights.items() 
            if metric in metrics
        )
        
        return score
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the evaluation results.
        
        Returns:
            Evaluation results
        """
        return self.evaluation_results
    
    def export_results(self, output_path: Optional[str] = None) -> str:
        """
        Export the evaluation results to a file.
        
        Args:
            output_path: Path to export results (defaults to evaluation_results.json in output_dir)
            
        Returns:
            Path to the exported results
        """
        if not output_path:
            output_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        
        with open(output_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results exported to {output_path}")
        
        return output_path
