# 🚀 Fine-tuning Qwen 1.7B - Pipeline Complète

## 📋 Vue d'ensemble

Ce repository contient une pipeline complète de fine-tuning pour le modèle Qwen 1.7B utilisant deux techniques : **GRPO** (Generative Reinforcement Learning from Policy Optimization) et **SFT** (Supervised Fine-Tuning) avec LoRA.

## 🎯 Objectifs

- **GRPO** : Le modèle prédit l'index du meilleur item et reçoit un reward basé sur le score
- **SFT** : Le modèle prédit l'index du meilleur item et reçoit un feedback correct/incorrect
- **Monitoring** : TensorBoard pour surveiller GPU, gradients et métriques
- **Checkpointing** : Sauvegarde fréquente (tous les 100-200 steps)

## 📁 Structure du Projet

```
Laforgue/
├── README.md                           # Ce fichier
├── README_finetuning.md               # Documentation technique
├── info_Antoine.md                    # Guide détaillé pour Antoine
├── GUIDE_ANTOINE_FINAL.md             # Guide complet pour Antoine
├── setup_antoine.sh                   # Script d'installation automatique
├── requirements_complete.txt           # Dépendances complètes
├── grpo_finetuning_v2.py             # Script GRPO (Version 2 recommandée)
├── sft_finetuning_v2.py              # Script SFT (Version 2 recommandée)
├── train_all_sectors_v2.sh           # Script de lancement v2
├── datasets_finaux/                   # Datasets de fine-tuning
│   ├── create_finetuning_datasets_v8.py
│   ├── dataset_adhesive_updated.json
│   ├── dataset_energy_submetering_updated.json
│   ├── dataset_full_flight_simulator_updated.json
│   ├── dataset_its_updated.json
│   ├── dataset_smart_water_updated.json
│   ├── dataset_solar_thermal_europe_updated.json
│   ├── dataset_submarine_drones_updated.json
│   └── dataset_waste_morocco_updated.json
└── .gitignore                         # Fichiers à ignorer
```

## 🚀 Installation Rapide

### Option 1 : Installation Automatique (Recommandée)

```bash
# Cloner le repository
git clone <URL_DU_REPO>
cd Laforgue

# Installation automatique
bash setup_antoine.sh
```

### Option 2 : Installation Manuelle

```bash
# Créer l'environnement conda
conda create -n qwen_finetuning python=3.10
conda activate qwen_finetuning

# Installer les dépendances
pip install -r requirements_complete.txt
```

## 🔧 Validation de la Pipeline

```bash
# Test rapide
python3 test_single_sector.py

# Validation complète
python3 validate_pipeline.py
```

## 🚀 Lancement du Fine-tuning

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

```bash
# Pour GRPO
tensorboard --logdir=./grpo_outputs --port=6006

# Pour SFT
tensorboard --logdir=./sft_outputs --port=6007
```

## 📊 Estimations de Ressources

### Par Secteur
- **VRAM** : ~2-3GB avec LoRA
- **Temps GRPO** : 2-4 heures
- **Temps SFT** : 1-2 heures
- **Total par secteur** : 3-6 heures

### Pour Tous les Secteurs
- **Temps total** : 24-48 heures
- **VRAM max** : ~3GB par modèle
- **Espace disque** : ~10GB pour les modèles finaux

## 🛡️ Garanties de Fonctionnement

### Tests de Validation Automatiques
- ✅ **Test de Pipeline Complète** : `python3 validate_pipeline.py`
- ✅ **Test de Secteur Unique** : `python3 test_single_sector.py`
- ✅ **Installation Automatique** : `bash setup_antoine.sh`

### Fonctionnalités Garanties
- ✅ **Cross Entropy Loss** : Parfaitement adaptée pour classification et prédiction
- ✅ **LoRA Configuration** : Optimisée pour Qwen 1.7B avec réduction VRAM
- ✅ **TensorBoard Monitoring** : GPU usage, gradients, loss, accuracy en temps réel
- ✅ **Checkpointing Fréquent** : Sauvegarde toutes les 100 steps
- ✅ **Instructions Réelles** : Chargement depuis `instructions_v6.json` avec fallback
- ✅ **Gestion d'Erreurs** : Logs détaillés et fallback robuste

## 📚 Documentation

- `info_Antoine.md` : Guide détaillé pour Antoine
- `GUIDE_ANTOINE_FINAL.md` : Guide complet pour Antoine
- `README_finetuning.md` : Documentation technique
- `requirements_complete.txt` : Dépendances complètes

## 🎯 Secteurs Disponibles

Le projet inclut 8 secteurs avec 200,000 échantillons au total :
- submarine_drones (25k échantillons)
- energy_submetering (25k échantillons)
- waste_morocco (25k échantillons)
- its (25k échantillons)
- smart_water (25k échantillons)
- adhesive (25k échantillons)
- solar_thermal_europe (25k échantillons)
- full_flight_simulator (25k échantillons)

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

## 📞 Support

Pour toute question ou problème :
1. Consulter la documentation dans `info_Antoine.md`
2. Vérifier les logs dans les dossiers `./grpo_outputs/` et `./sft_outputs/`
3. Utiliser TensorBoard pour le monitoring en temps réel

---

**🎉 Prêt pour le fine-tuning sur cluster GPU ! 🚀**
