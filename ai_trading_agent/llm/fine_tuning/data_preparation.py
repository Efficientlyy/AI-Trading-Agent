"""
Financial data preparation for LLM fine-tuning.

This module provides tools for collecting, cleaning, and processing
financial domain data for fine-tuning language models.
"""
import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .config import DatasetType, FineTuningConfig


logger = logging.getLogger(__name__)


class FinancialDatasetPreparer:
    """
    Prepares datasets for fine-tuning LLMs on financial domain knowledge.
    
    This class handles the collection, cleaning, and formatting of various
    financial datasets for model fine-tuning.
    """
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the dataset preparer with configuration.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.datasets = {}
        self.processed_data = {
            "train": [],
            "validation": [],
            "test": []
        }
    
    def collect_datasets(self) -> Dict[DatasetType, pd.DataFrame]:
        """
        Collect all datasets specified in the configuration.
        
        Returns:
            Dictionary mapping dataset types to their DataFrames
        """
        for dataset_type in self.config.dataset_types:
            if dataset_type in self.config.dataset_paths:
                path = self.config.dataset_paths[dataset_type]
                self.datasets[dataset_type] = self._load_dataset(dataset_type, path)
            else:
                logger.warning(f"No path specified for dataset type {dataset_type.value}")
        
        return self.datasets
    
    def _load_dataset(self, dataset_type: DatasetType, path: str) -> pd.DataFrame:
        """
        Load a dataset from disk.
        
        Args:
            dataset_type: Type of dataset to load
            path: Path to the dataset file or directory
            
        Returns:
            DataFrame containing the loaded dataset
        """
        logger.info(f"Loading {dataset_type.value} dataset from {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        
        if os.path.isdir(path):
            # Handle directory of files
            return self._load_directory(dataset_type, path)
        else:
            # Handle single file
            file_ext = os.path.splitext(path)[1].lower()
            
            if file_ext == '.csv':
                return pd.read_csv(path)
            elif file_ext == '.jsonl' or file_ext == '.json':
                return self._load_json_dataset(path)
            elif file_ext == '.txt':
                return self._load_text_dataset(path, dataset_type)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
    
    def _load_directory(self, dataset_type: DatasetType, path: str) -> pd.DataFrame:
        """
        Load all files in a directory into a single DataFrame.
        
        Args:
            dataset_type: Type of dataset being loaded
            path: Directory path
            
        Returns:
            Combined DataFrame
        """
        all_data = []
        
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                try:
                    if file_ext == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_ext == '.jsonl' or file_ext == '.json':
                        df = self._load_json_dataset(file_path)
                    elif file_ext == '.txt':
                        df = self._load_text_dataset(file_path, dataset_type)
                    elif file_ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path)
                    else:
                        logger.warning(f"Skipping unsupported file type: {file}")
                        continue
                    
                    # Add source file information
                    df['source_file'] = file
                    all_data.append(df)
                    
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
        
        if not all_data:
            raise ValueError(f"No valid files found in directory: {path}")
        
        return pd.concat(all_data, ignore_index=True)
    
    def _load_json_dataset(self, path: str) -> pd.DataFrame:
        """
        Load a JSON or JSONL dataset.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            DataFrame with loaded data
        """
        if path.endswith('.jsonl'):
            # Load JSON Lines file
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            # Load regular JSON file
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict):
                    # Handle the case of a JSON object
                    if 'data' in content:
                        data = content['data']
                    else:
                        # Convert single dict to a list with one item
                        data = [content]
                else:
                    raise ValueError(f"Unexpected JSON structure in {path}")
        
        return pd.DataFrame(data)
    
    def _load_text_dataset(self, path: str, dataset_type: DatasetType) -> pd.DataFrame:
        """
        Load a plain text dataset.
        
        Args:
            path: Path to the text file
            dataset_type: Type of dataset for context-appropriate parsing
            
        Returns:
            DataFrame with text data
        """
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Different parsing strategies based on dataset type
        if dataset_type == DatasetType.REGULATORY_DOCS:
            return self._parse_regulatory_doc(text, path)
        elif dataset_type == DatasetType.MARKET_COMMENTARY:
            return self._parse_market_commentary(text, path)
        elif dataset_type == DatasetType.FINANCIAL_NEWS:
            return self._parse_financial_news(text, path)
        elif dataset_type == DatasetType.SEC_FILINGS:
            return self._parse_sec_filing(text, path)
        else:
            # Generic parsing for other types
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return pd.DataFrame({
                'text': paragraphs,
                'source': os.path.basename(path)
            })
    
    def _parse_regulatory_doc(self, text: str, path: str) -> pd.DataFrame:
        """Parse regulatory document format."""
        # Split by sections (usually indicated by headers)
        section_pattern = r'(#{1,6}\s+.+|\b[A-Z][A-Z\s]+:|\b[IVX]+\.\s+.+|\b\d+\.\d+\s+.+)'
        sections = re.split(section_pattern, text)
        
        if len(sections) <= 1:
            # Fallback if no sections found
            sections = [p.strip() for p in text.split('\n\n') if p.strip()]
            return pd.DataFrame({
                'text': sections,
                'source': os.path.basename(path),
                'type': 'regulatory',
                'section': 'undefined'
            })
        
        # Combine headers with their content
        processed_sections = []
        for i in range(0, len(sections)-1, 2):
            if i+1 < len(sections):
                header = sections[i].strip()
                content = sections[i+1].strip()
                if header and content:
                    processed_sections.append({
                        'text': content,
                        'source': os.path.basename(path),
                        'type': 'regulatory',
                        'section': header
                    })
        
        return pd.DataFrame(processed_sections)
    
    def _parse_market_commentary(self, text: str, path: str) -> pd.DataFrame:
        """Parse market commentary format."""
        # Typically date-centered entries
        date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        entries = re.split(date_pattern, text)
        
        if len(entries) <= 1:
            # Fallback if no date patterns found
            entries = [p.strip() for p in text.split('\n\n') if p.strip()]
            return pd.DataFrame({
                'text': entries,
                'source': os.path.basename(path),
                'type': 'commentary',
                'date': None
            })
        
        # Extract dates and their content
        processed_entries = []
        for i in range(1, len(entries)):
            if i < len(entries):
                # Find the date that precedes this entry
                if i > 1:
                    date_match = re.search(date_pattern, text.split(entries[i-1])[-1] + entries[i-1])
                else:
                    date_match = re.search(date_pattern, text.split(entries[i])[0])
                
                date_str = date_match.group(0) if date_match else None
                content = entries[i].strip()
                
                if content:
                    try:
                        date_obj = pd.to_datetime(date_str) if date_str else None
                        date_iso = date_obj.isoformat() if date_obj else None
                    except:
                        date_iso = None
                        
                    processed_entries.append({
                        'text': content,
                        'source': os.path.basename(path),
                        'type': 'commentary',
                        'date': date_iso
                    })
        
        return pd.DataFrame(processed_entries)
    
    def _parse_financial_news(self, text: str, path: str) -> pd.DataFrame:
        """Parse financial news format."""
        # News often has headlines/title separated from body
        lines = text.split('\n')
        articles = []
        current_article = {"headline": "", "body": []}
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a new headline (usually short and ends with newlines)
            if not line and current_article["body"]:
                # Save current article if it has content
                if current_article["headline"] or current_article["body"]:
                    articles.append({
                        'headline': current_article["headline"],
                        'text': '\n'.join(current_article["body"]),
                        'source': os.path.basename(path),
                        'type': 'news'
                    })
                # Start new article
                current_article = {"headline": "", "body": []}
            elif not current_article["body"] and line and not current_article["headline"]:
                # This is likely a headline
                current_article["headline"] = line
            elif line:
                # Add to the body
                current_article["body"].append(line)
        
        # Add the last article if not empty
        if current_article["headline"] or current_article["body"]:
            articles.append({
                'headline': current_article["headline"],
                'text': '\n'.join(current_article["body"]),
                'source': os.path.basename(path),
                'type': 'news'
            })
        
        if not articles:
            # Fallback if no articles parsed
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return pd.DataFrame({
                'text': paragraphs,
                'source': os.path.basename(path),
                'type': 'news',
                'headline': ''
            })
        
        return pd.DataFrame(articles)
    
    def _parse_sec_filing(self, text: str, path: str) -> pd.DataFrame:
        """Parse SEC filing format."""
        # SEC filings often have item numbers and section headers
        item_pattern = r'(ITEM\s+\d+\..*?)(?=ITEM\s+\d+\.|$)'
        items = re.findall(item_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not items:
            # Fallback if no items found
            sections = [p.strip() for p in text.split('\n\n') if p.strip()]
            return pd.DataFrame({
                'text': sections,
                'source': os.path.basename(path),
                'type': 'sec_filing',
                'item': 'undefined'
            })
        
        processed_items = []
        for item in items:
            # Extract item header and content
            item_match = re.match(r'(ITEM\s+\d+\..*?)(?=\n)', item, re.IGNORECASE)
            
            if item_match:
                header = item_match.group(1).strip()
                content = item[len(header):].strip()
                
                processed_items.append({
                    'text': content,
                    'source': os.path.basename(path),
                    'type': 'sec_filing',
                    'item': header
                })
            else:
                processed_items.append({
                    'text': item.strip(),
                    'source': os.path.basename(path),
                    'type': 'sec_filing',
                    'item': 'undefined'
                })
        
        return pd.DataFrame(processed_items)
    
    def prepare_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare the collected datasets for fine-tuning.
        
        This method cleans, formats, and splits the data into train,
        validation, and test sets.
        
        Returns:
            Dictionary with train, validation, and test datasets
        """
        if not self.datasets:
            # Collect datasets if not already done
            self.collect_datasets()
        
        # Process each dataset
        for dataset_type, df in self.datasets.items():
            processed = self._process_dataset(dataset_type, df)
            
            # Split into train, validation, test
            train, val_test = train_test_split(
                processed, 
                test_size=self.config.validation_split + self.config.test_split,
                random_state=42
            )
            
            # Further split val_test into validation and test
            val_ratio = self.config.validation_split / (self.config.validation_split + self.config.test_split)
            validation, test = train_test_split(
                val_test, 
                test_size=1-val_ratio,
                random_state=42
            )
            
            # Add to respective splits
            self.processed_data["train"].extend(train)
            self.processed_data["validation"].extend(validation)
            self.processed_data["test"].extend(test)
        
        return self.processed_data
