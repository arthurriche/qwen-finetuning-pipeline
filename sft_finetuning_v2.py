#!/usr/bin/env python3
"""
Script de fine-tuning SFT pour Qwen 1.7B avec LoRA et Cross Entropy Loss
"""

import os
import json
import torch
import logging
import numpy as np
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_instructions(sector):
    """Charge les vraies instructions sectorielles depuis instructions_v6.json"""
    try:
        with open("instructions_v6.json", 'r', encoding='utf-8') as f:
            instructions = json.load(f)
            instruction = instructions.get(sector)
            if instruction:
                logger.info(f"✅ Instructions chargées pour le secteur: {sector}")
                return instruction
            else:
                logger.warning(f"⚠️ Secteur {sector} non trouvé dans instructions_v6.json")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des instructions: {e}")
    
    # Fallback vers les instructions simplifiées
    logger.info("🔄 Utilisation des instructions de fallback")
    instruction_mapping = {
        "submarine_drones": "You are an expert in submarine drones and underwater robotics. Given 5 options, choose the one that would be most relevant for finding submarine drone manufacturers and suppliers.",
        "energy_submetering": "You are an expert in energy submetering systems. Given 5 options, choose the one that would be most relevant for finding energy submetering companies and solutions.",
        "waste_morocco": "You are an expert in hazardous waste treatment in Morocco. Given 5 options, choose the one that would be most relevant for finding waste treatment companies in Morocco.",
        "its": "You are an expert in intelligent transport systems. Given 5 options, choose the one that would be most relevant for finding ITS companies and solutions.",
        "smart_water": "You are an expert in smart water services. Given 5 options, choose the one that would be most relevant for finding smart water companies and solutions.",
        "adhesive": "You are an expert in adhesive tape converters in Belgium. Given 5 options, choose the one that would be most relevant for finding adhesive tape companies in Belgium.",
        "solar_thermal_europe": "You are an expert in solar thermal systems in Europe. Given 5 options, choose the one that would be most relevant for finding solar thermal companies in Europe.",
        "full_flight_simulator": "You are an expert in full flight simulators. Given 5 options, choose the one that would be most relevant for finding flight simulator manufacturers."
    }
    return instruction_mapping.get(sector, "Given 5 options, choose the most relevant one.")

def prepare_dataset(data_path, sector):
    """Prépare le dataset pour l'entraînement SFT avec vraies instructions"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    instruction = load_instructions(sector)
    
    processed_data = []
    for item in data:
        # Créer le prompt avec vraie instruction
        prompt = f"{instruction}\n\nOptions:\n{item["input"]}\n\nAnswer (1-5): "
        target_idx = int(item['output']) - 1
        
        # Créer la réponse attendue (format SFT)
        response = f"{target_idx + 1}"  # Retourner 1-5
        
        # Combiner prompt + réponse pour SFT
        full_text = prompt + response
        
        processed_data.append({
            "text": full_text,
            "target": target_idx,
            "scores": [float(s) for s in item['score'].split(',')],
            "best_score": float(item['score'].split(',')[target_idx])
        })
    
    logger.info(f"📊 Dataset préparé: {len(processed_data)} échantillons")
    return Dataset.from_list(processed_data)

def create_lora_config():
    """Configuration LoRA"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

class SFTTrainer(Trainer):
    """Trainer personnalisé pour SFT avec Cross Entropy Loss"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=f"./sft_outputs/{kwargs.get('sector', 'default')}/tensorboard")
        self.step_count = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Calcul de la loss avec Cross Entropy pour SFT"""
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Pour SFT, on prédit la réponse complète
        # On utilise la loss standard de causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        # Calculer la loss
        loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Logging pour TensorBoard
        if self.step_count % 10 == 0:
            with torch.no_grad():
                # Calculer la perplexité
                perplexity = torch.exp(loss)
                
                # Log des métriques
                self.writer.add_scalar('Loss/train', loss.item(), self.step_count)
                self.writer.add_scalar('Perplexity/train', perplexity.item(), self.step_count)
                
                # GPU monitoring
                if torch.cuda.is_available():
                    self.writer.add_scalar('GPU/memory_allocated', torch.cuda.memory_allocated() / 1024**3, self.step_count)
                    self.writer.add_scalar('GPU/memory_reserved', torch.cuda.memory_reserved() / 1024**3, self.step_count)
        
        self.step_count += 1
        
        return (loss, outputs) if return_outputs else loss

def main():
    # Configuration
    model_path = "./models/qwen1.7b"
    sector = "submarine_drones"  # Changer selon le secteur
    data_path = f"finetuning_datasets_v8/{sector}/unified_training_data_v8.json"
    
    # Vérifier GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Charger le modèle et tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configuration LoRA
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Préparer le dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(data_path, sector)
    
    # Configuration d'entraînement
    training_args = TrainingArguments(
        output_dir=f"./sft_outputs/{sector}",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_steps=500,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=4,
        save_total_limit=3,  # Garder seulement 3 checkpoints
    )
    
    # Tokenizer function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # Tokenizer le dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer personnalisé
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        sector=sector
    )
    
    # Lancer l'entraînement
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Sauvegarder le modèle
    trainer.save_model(f"./sft_outputs/{sector}/final_model")
    logger.info("SFT training completed!")
    
    # Fermer TensorBoard writer
    trainer.writer.close()

if __name__ == "__main__":
    main() 