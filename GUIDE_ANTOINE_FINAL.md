# 🚀 Guide Complet pour Antoine - Fine-tuning Qwen 1.7B

## 📋 Vue d'ensemble

Ce guide contient toutes les étapes pour lancer le fine-tuning de Qwen 1.7B avec LoRA sur votre cluster GPU. Le projet utilise deux techniques : **GRPO** et **SFT** avec Cross Entropy Loss.

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

## 🚀 Installation Automatique

### Option 1 : Installation Automatique (Recommandée)

```bash
# 1. Cloner ou télécharger le projet
# 2. Aller dans le dossier du projet
cd /chemin/vers/le/projet

# 3. Lancer l'installation automatique
bash setup_antoine.sh
```

Le script `setup_antoine.sh` va :
- ✅ Vérifier Python et pip
- ✅ Détecter les GPU NVIDIA
- ✅ Créer un environnement virtuel
- ✅ Installer toutes les dépendances
- ✅ Télécharger le modèle Qwen 1.7B (optionnel)
- ✅ Valider la pipeline
- ✅ Tester les composants

### Option 2 : Installation Manuelle

```bash
# 1. Créer l'environnement conda
conda create -n qwen_finetuning python=3.10
conda activate qwen_finetuning

# 2. Installer les dépendances
pip install -r requirements_complete.txt

# 3. Télécharger le modèle (optionnel)
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
cd ..
```

## 🔧 Validation de la Pipeline

### Test Rapide
```bash
python3 test_single_sector.py
```

### Validation Complète
```bash
python3 validate_pipeline.py
```

Ces scripts vérifient :
- ✅ Chargement des datasets
- ✅ Chargement des instructions
- ✅ Génération des prompts
- ✅ Syntaxe des scripts
- ✅ Fichiers de configuration

## 🚀 Lancement du Fine-tuning

### Scripts Disponibles

#### Version 1 (Instructions Simplifiées)
- `grpo_finetuning.py` : Script GRPO avec instructions simplifiées
- `sft_finetuning.py` : Script SFT avec instructions simplifiées
- `train_all_sectors.sh` : Script de lancement

#### Version 2 (Instructions Réelles) ⭐ RECOMMANDÉE
- `grpo_finetuning_v2.py` : Script GRPO avec vraies instructions sectorielles
- `sft_finetuning_v2.py` : Script SFT avec vraies instructions sectorielles
- `train_all_sectors_v2.sh` : Script de lancement v2

### Lancement d'un Secteur Spécifique

```bash
# Activer l'environnement
source venv_finetuning/bin/activate  # ou conda activate qwen_finetuning

# GRPO (Version 2 recommandée)
python grpo_finetuning_v2.py

# SFT (Version 2 recommandée)
python sft_finetuning_v2.py
```

### Lancement de Tous les Secteurs

```bash
# Version 2 (Recommandée)
bash train_all_sectors_v2.sh
```

## 📊 Monitoring

### TensorBoard
```bash
# Pour GRPO
tensorboard --logdir=./grpo_outputs --port=6006

# Pour SFT
tensorboard --logdir=./sft_outputs --port=6007
```

### Métriques Surveillées
- **Loss** : Cross Entropy Loss
- **Accuracy** : Précision de classification (GRPO)
- **Perplexity** : Perplexité du modèle (SFT)
- **GPU Usage** : Utilisation VRAM et mémoire
- **Gradients** : Normes des gradients

## 💾 Gestion des Ressources

### Estimations de Ressources
- **VRAM par modèle** : ~2-3GB avec LoRA
- **Batch size** : 4-8 selon la VRAM
- **Gradient accumulation** : 4-8 steps
- **Temps par secteur** : 2-4h (GRPO) + 1-2h (SFT) = 3-6h total
- **Temps total pour 8 secteurs** : 24-48 heures

### Optimisations
- Utiliser `fp16=True` pour réduire l'utilisation mémoire
- Ajuster `per_device_train_batch_size` selon la VRAM disponible
- Utiliser `gradient_accumulation_steps` pour simuler des batchs plus grands

## 🔧 Dépannage

### Erreurs Communes

1. **CUDA out of memory**
   ```bash
   # Réduire le batch size dans les scripts
   per_device_train_batch_size=2  # au lieu de 4
   ```

2. **ModuleNotFoundError**
   ```bash
   # Réinstaller les dépendances
   pip install -r requirements_complete.txt
   ```

3. **SyntaxError dans les scripts**
   ```bash
   # Vérifier la syntaxe
   python3 test_single_sector.py
   ```

### Commandes Utiles

```bash
# Vérifier l'utilisation GPU
watch -n 1 nvidia-smi

# Vérifier les processus Python
ps aux | grep python

# Nettoyer les checkpoints
find . -name "checkpoint-*" -type d -exec rm -rf {} +

# Vérifier l'espace disque
df -h
```

## 📈 Résultats Attendus

### Métriques de Succès
- **Loss** : Décroissante au fil du temps
- **Accuracy** : Croissante vers 0.8-0.9
- **Perplexity** : Décroissante vers 1.0-2.0
- **GPU Usage** : Stable et optimisé

### Fichiers de Sortie
- **Modèles finaux** : `./grpo_outputs/{sector}/final_model/`
- **Logs TensorBoard** : `./grpo_outputs/{sector}/tensorboard/`
- **Checkpoints** : `./grpo_outputs/{sector}/checkpoint-*/`

## 📞 Support

### En Cas de Problème
1. Vérifier les logs dans les dossiers `./grpo_outputs/` et `./sft_outputs/`
2. Consulter TensorBoard pour les métriques en temps réel
3. Vérifier l'utilisation GPU avec `nvidia-smi`
4. Contacter Arthur pour assistance technique

### Documentation Disponible
- `info_Antoine.md` : Guide détaillé
- `README_finetuning.md` : Documentation technique
- `VERSIONS_COMPARISON.md` : Comparaison des versions
- `VERSION_V2_INFO.md` : Info sur les scripts v2
- `RESUME_FINAL.md` : Résumé du projet

## 🎯 Checklist de Lancement

- [ ] Environnement virtuel créé et activé
- [ ] Dépendances installées (`pip install -r requirements_complete.txt`)
- [ ] Tests de validation passés (`python3 test_single_sector.py`)
- [ ] GPU détecté et disponible
- [ ] Espace disque suffisant (>50GB)
- [ ] TensorBoard lancé pour monitoring
- [ ] Scripts de fine-tuning prêts

## 🚀 Commandes de Lancement Finales

```bash
# 1. Installation automatique
bash setup_antoine.sh

# 2. Activation de l'environnement
source venv_finetuning/bin/activate

# 3. Validation
python3 test_single_sector.py

# 4. Lancement du fine-tuning
bash train_all_sectors_v2.sh

# 5. Monitoring (dans un autre terminal)
tensorboard --logdir=./grpo_outputs --port=6006
tensorboard --logdir=./sft_outputs --port=6007
```

---

**Bonne chance avec le fine-tuning ! 🚀**

Le projet est prêt pour l'exécution sur votre cluster GPU.
