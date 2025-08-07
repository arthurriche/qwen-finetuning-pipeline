# Fine-tuning Qwen 1.7B avec LoRA - GRPO et SFT

## 🎯 Objectif

Ce projet implémente le fine-tuning de Qwen 1.7B avec LoRA en utilisant deux techniques :
- **GRPO** : Le modèle prédit l'index du meilleur item et reçoit un reward basé sur le score
- **SFT** : Le modèle prédit l'index du meilleur item et reçoit un feedback correct/incorrect

## 📊 Datasets

Les datasets de fine-tuning sont générés à partir de 8 secteurs différents :
- submarine_drones (25k échantillons)
- energy_submetering (25k échantillons)
- waste_morocco (25k échantillons)
- its (25k échantillons)
- smart_water (25k échantillons)
- adhesive (25k échantillons)
- solar_thermal_europe (25k échantillons)
- full_flight_simulator (25k échantillons)

**Total : 200,000 échantillons**

## 🚀 Installation Rapide

```bash
# 1. Créer l'environnement
conda create -n qwen_finetuning python=3.10
conda activate qwen_finetuning

# 2. Installer les dépendances
pip install -r requirements_finetuning.txt

# 3. Télécharger Qwen 1.7B
mkdir -p models
cd models
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

## 📁 Structure des Fichiers

```
├── finetuning_datasets_v8/          # Datasets générés
│   ├── submarine_drones/
│   ├── energy_submetering/
│   └── ...
├── grpo_finetuning.py              # Script GRPO
├── sft_finetuning.py               # Script SFT
├── train_all_sectors.sh            # Script de lancement
├── info_Antoine.md                 # Guide complet
├── requirements_finetuning.txt      # Dépendances
└── README_finetuning.md            # Ce fichier
```

## 🔧 Configuration

### Cross Entropy Loss
- **GRPO** : Classification des 5 options avec Cross Entropy Loss
- **SFT** : Prédiction de la réponse complète avec Cross Entropy Loss

### LoRA Configuration
```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

### Monitoring TensorBoard
- Loss et accuracy en temps réel
- Utilisation GPU et mémoire
- Gradients et métriques d'entraînement
- Checkpoints toutes les 100 steps

## 🚀 Lancement

### Fine-tuning d'un secteur spécifique

```bash
# GRPO
python grpo_finetuning.py

# SFT
python sft_finetuning.py
```

### Fine-tuning de tous les secteurs

```bash
bash train_all_sectors.sh
```

### Monitoring

```bash
# TensorBoard GRPO
tensorboard --logdir=./grpo_outputs --port=6006

# TensorBoard SFT
tensorboard --logdir=./sft_outputs --port=6007
```

## 📈 Estimations de Ressources

### Par Secteur
- **VRAM** : ~2-3GB avec LoRA
- **Temps GRPO** : 2-4 heures
- **Temps SFT** : 1-2 heures
- **Total par secteur** : 3-6 heures

### Pour Tous les Secteurs
- **Temps total** : 24-48 heures
- **VRAM max** : ~3GB par modèle
- **Espace disque** : ~10GB pour les modèles finaux

### Optimisations
- `fp16=True` pour réduire l'utilisation mémoire
- `gradient_accumulation_steps=4` pour simuler des batchs plus grands
- `save_total_limit=3` pour limiter l'espace disque

## 📊 Métriques Surveillées

### GRPO
- Loss Cross Entropy
- Accuracy de classification
- GPU memory usage
- Gradient norms

### SFT
- Loss Cross Entropy
- Perplexity
- GPU memory usage
- Gradient norms

## 🔍 Dépannage

### Erreurs communes
1. **CUDA out of memory** : Réduire `per_device_train_batch_size`
2. **Tokenization errors** : Vérifier la longueur des prompts
3. **Slow training** : Augmenter `dataloader_num_workers`

### Commandes utiles
```bash
# Vérifier GPU
nvidia-smi

# Vérifier PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Nettoyer les checkpoints
find . -name "checkpoint-*" -type d -exec rm -rf {} +
```

## 📞 Support

Pour plus d'informations, consulter :
- `info_Antoine.md` : Guide détaillé pour Antoine
- Logs dans `./grpo_outputs/` et `./sft_outputs/`
- TensorBoard pour les métriques en temps réel

---

**Bonne chance avec le fine-tuning ! 🚀** 