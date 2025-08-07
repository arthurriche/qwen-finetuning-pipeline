# Guide de Fine-tuning Qwen 1.7B - Instructions pour Antoine

## 📋 Vue d'ensemble

Ce guide contient toutes les étapes nécessaires pour lancer le fine-tuning de Qwen 1.7B avec LoRA sur votre cluster GPU. Le projet utilise deux techniques : **GRPO** et **SFT** avec Cross Entropy Loss.

## 🎯 Objectifs

- **GRPO** : Le modèle prédit l'index du meilleur item et reçoit un reward basé sur le score
- **SFT** : Le modèle prédit l'index du meilleur item et reçoit un feedback correct/incorrect
- **Monitoring** : TensorBoard pour surveiller GPU, gradients et métriques
- **Checkpointing** : Sauvegarde fréquente (tous les 100-200 steps)

## 📁 Structure des données

Les datasets sont dans le dossier `finetuning_datasets_v8/` avec 8 secteurs :
- submarine_drones (25k échantillons)
- energy_submetering (25k échantillons)
- waste_morocco (25k échantillons)
- its (25k échantillons)
- smart_water (25k échantillons)
- adhesive (25k échantillons)
- solar_thermal_europe (25k échantillons)
- full_flight_simulator (25k échantillons)

**Total : 200,000 échantillons**

## 🚀 Installation et Setup

### 1. Créer l'environnement conda

```bash
# Créer un nouvel environnement
conda create -n qwen_finetuning python=3.10
conda activate qwen_finetuning

# Installer les dépendances
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install peft==0.7.1
pip install accelerate==0.25.0
pip install datasets==2.14.0
pip install tensorboard==2.15.0
pip install wandb==0.16.0
pip install scikit-learn==1.3.0
pip install numpy==1.24.3
pip install tqdm==4.66.0
```

### 2. Vérifier l'installation CUDA

```bash
# Vérifier CUDA
nvidia-smi

# Vérifier PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 3. Télécharger le modèle Qwen 1.7B

```bash
# Créer le dossier pour les modèles
mkdir -p models
cd models

# Télécharger Qwen 1.7B (peut prendre du temps)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'Qwen/Qwen1.5-1.7B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
tokenizer.save_pretrained('./qwen1.7b')
model.save_pretrained('./qwen1.7b')
print('Modèle téléchargé avec succès!')
"
```

## 📊 Structure des données d'entraînement

Chaque échantillon contient :
```json
{
  "input": "1. item1\n2. item2\n3. item3\n4. item4\n5. item5",
  "output": "3",  // Index du meilleur item (0-4)
  "score": "9,1,12,12,7",  // Scores bruts séparés par virgules
  "type": "keyword"  // ou "url"
}
```

## 🔧 Scripts de Fine-tuning

### Script GRPO (grpo_finetuning.py)

```python
#!/usr/bin/env python3
"""
Script de fine-tuning GRPO pour Qwen 1.7B avec LoRA
"""

import os
import json
import torch
import logging
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

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_instructions(sector):
    """Charge les instructions sectorielles"""
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
    """Prépare le dataset pour l'entraînement"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    instruction = load_instructions(sector)
    
    processed_data = []
    for item in data:
        # Créer le prompt avec instruction
        prompt = f"{instruction}\n\nOptions:\n{item['input']}\n\nAnswer (1-5): "
        
        # Convertir l'output en index (1-5 -> 0-4)
        target_idx = int(item['output']) - 1
        
        processed_data.append({
            "text": prompt,
            "target": target_idx,
            "scores": item['score']
        })
    
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
        output_dir=f"./grpo_outputs/{sector}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
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
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Lancer l'entraînement
    logger.info("Starting training...")
    trainer.train()
    
    # Sauvegarder le modèle
    trainer.save_model(f"./grpo_outputs/{sector}/final_model")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
```

### Script SFT (sft_finetuning.py)

```python
#!/usr/bin/env python3
"""
Script de fine-tuning SFT pour Qwen 1.7B avec LoRA
"""

import os
import json
import torch
import logging
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

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_instructions(sector):
    """Charge les instructions sectorielles"""
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
    """Prépare le dataset pour l'entraînement SFT"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    instruction = load_instructions(sector)
    
    processed_data = []
    for item in data:
        # Créer le prompt avec instruction
        prompt = f"{instruction}\n\nOptions:\n{item['input']}\n\nAnswer (1-5): "
        
        # Convertir l'output en index (1-5 -> 0-4)
        target_idx = int(item['output']) - 1
        
        # Créer la réponse attendue
        response = f"{target_idx + 1}"  # Retourner 1-5
        
        processed_data.append({
            "text": prompt + response,
            "target": target_idx,
            "scores": item['score']
        })
    
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
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Lancer l'entraînement
    logger.info("Starting training...")
    trainer.train()
    
    # Sauvegarder le modèle
    trainer.save_model(f"./sft_outputs/{sector}/final_model")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
```

## 🚀 Commandes de lancement

### 1. Lancer TensorBoard (dans un terminal séparé)

```bash
# Pour GRPO
tensorboard --logdir=./grpo_outputs --port=6006

# Pour SFT  
tensorboard --logdir=./sft_outputs --port=6007
```

### 2. Lancer le fine-tuning GRPO

```bash
# Pour un secteur spécifique
python grpo_finetuning.py

# Ou modifier le secteur dans le script et relancer
```

### 3. Lancer le fine-tuning SFT

```bash
# Pour un secteur spécifique
python sft_finetuning.py

# Ou modifier le secteur dans le script et relancer
```

### 4. Script de lancement pour tous les secteurs

```bash
#!/bin/bash
# train_all_sectors.sh

sectors=("submarine_drones" "energy_submetering" "waste_morocco" "its" "smart_water" "adhesive" "solar_thermal_europe" "full_flight_simulator")

for sector in "${sectors[@]}"; do
    echo "Training GRPO for sector: $sector"
    sed -i "s/sector = \".*\"/sector = \"$sector\"/" grpo_finetuning.py
    python grpo_finetuning.py
    
    echo "Training SFT for sector: $sector"
    sed -i "s/sector = \".*\"/sector = \"$sector\"/" sft_finetuning.py
    python sft_finetuning.py
done
```

## 📊 Monitoring et Logs

### TensorBoard
- Accéder à `http://localhost:6006` pour GRPO
- Accéder à `http://localhost:6007` pour SFT
- Métriques disponibles : loss, accuracy, GPU usage, gradients

### Logs
- Les logs sont sauvegardés dans `./grpo_outputs/` et `./sft_outputs/`
- Checkpoints toutes les 100 steps
- Validation toutes les 500 steps

## 💾 Gestion des ressources

### Estimations de ressources
- **VRAM par modèle** : ~2-3GB avec LoRA
- **Batch size** : 4-8 selon la VRAM
- **Gradient accumulation** : 4-8 steps
- **Temps estimé par secteur** : 2-4 heures (GRPO) + 1-2 heures (SFT)

### Optimisations
- Utiliser `fp16=True` pour réduire l'utilisation mémoire
- Ajuster `per_device_train_batch_size` selon la VRAM disponible
- Utiliser `gradient_accumulation_steps` pour simuler des batchs plus grands

## 🔧 Dépannage

### Erreurs communes
1. **CUDA out of memory** : Réduire `per_device_train_batch_size`
2. **Tokenization errors** : Vérifier la longueur des prompts
3. **Slow training** : Augmenter `dataloader_num_workers`

### Commandes utiles
```bash
# Vérifier l'utilisation GPU
watch -n 1 nvidia-smi

# Vérifier les processus Python
ps aux | grep python

# Nettoyer les checkpoints
find . -name "checkpoint-*" -type d -exec rm -rf {} +
```

## 📈 Résultats attendus

- **GRPO** : Modèle qui prédit l'index du meilleur item avec reward
- **SFT** : Modèle qui prédit l'index du meilleur item avec feedback
- **Métriques** : Loss décroissante, accuracy croissante
- **Temps** : 3-6 heures par secteur complet

## 📞 Support

En cas de problème :
1. Vérifier les logs dans les dossiers `./grpo_outputs/` et `./sft_outputs/`
2. Consulter TensorBoard pour les métriques
3. Vérifier l'utilisation GPU avec `nvidia-smi`
4. Contacter Arthur pour assistance technique

---

**Bonne chance avec le fine-tuning ! 🚀** 