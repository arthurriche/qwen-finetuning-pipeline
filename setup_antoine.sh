#!/bin/bash
# Script d'installation automatique pour Antoine
# Usage: bash setup_antoine.sh

set -e  # Arrêter en cas d'erreur

echo "🚀 Installation automatique de la pipeline de fine-tuning"
echo "⏰ Début: $(date)"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour les messages colorés
log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Vérifier Python
echo ""
echo "🔍 Vérification de Python..."
python_version=$(python3 --version 2>&1)
if [[ $python_version == Python* ]]; then
    log_success "Python détecté: $python_version"
else
    log_error "Python3 non trouvé. Veuillez installer Python 3.8+"
    exit 1
fi

# 2. Vérifier pip
echo ""
echo "🔍 Vérification de pip..."
if command -v pip3 &> /dev/null; then
    log_success "pip3 disponible"
else
    log_error "pip3 non trouvé. Veuillez installer pip"
    exit 1
fi

# 3. Vérifier CUDA
echo ""
echo "🔍 Vérification de CUDA..."
if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA GPU détecté"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    log_warning "NVIDIA GPU non détecté. Le fine-tuning sera plus lent sur CPU"
fi

# 4. Créer l'environnement virtuel
echo ""
echo "🔧 Création de l'environnement virtuel..."
if [ -d "venv_finetuning" ]; then
    log_warning "Environnement virtuel existant détecté"
    read -p "Voulez-vous le supprimer et en créer un nouveau ? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv_finetuning
        log_success "Ancien environnement supprimé"
    fi
fi

if [ ! -d "venv_finetuning" ]; then
    python3 -m venv venv_finetuning
    log_success "Environnement virtuel créé"
fi

# 5. Activer l'environnement
echo ""
echo "🔧 Activation de l'environnement virtuel..."
source venv_finetuning/bin/activate
log_success "Environnement activé"

# 6. Mettre à jour pip
echo ""
echo "🔧 Mise à jour de pip..."
pip install --upgrade pip
log_success "pip mis à jour"

# 7. Installer les dépendances
echo ""
echo "�� Installation des dépendances..."
if [ -f "requirements_complete.txt" ]; then
    pip install -r requirements_complete.txt
    log_success "Dépendances installées"
else
    log_error "Fichier requirements_complete.txt manquant"
    exit 1
fi

# 8. Vérifier l'installation
echo ""
echo "🔍 Vérification de l'installation..."
python3 -c "
import torch
import transformers
import peft
import datasets
print('✅ Toutes les dépendances sont installées')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'Datasets: {datasets.__version__}')
"

# 9. Télécharger le modèle (optionnel)
echo ""
echo "🔧 Téléchargement du modèle Qwen 1.7B..."
read -p "Voulez-vous télécharger le modèle maintenant ? (peut prendre du temps) (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p models
    cd models
    python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Téléchargement du tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-1.7B')
print('Téléchargement du modèle...')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-1.7B', torch_dtype='auto', device_map='auto')
print('Sauvegarde du modèle...')
tokenizer.save_pretrained('./qwen1.7b')
model.save_pretrained('./qwen1.7b')
print('✅ Modèle téléchargé avec succès!')
"
    cd ..
    log_success "Modèle téléchargé"
else
    log_warning "Modèle non téléchargé. Il sera téléchargé automatiquement lors du premier lancement."
fi

# 10. Validation de la pipeline
echo ""
echo "🔍 Validation de la pipeline..."
python3 validate_pipeline.py

# 11. Test de la pipeline
echo ""
echo "🔍 Test de la pipeline..."
python3 test_single_sector.py

echo ""
echo "🎉 Installation terminée !"
echo "⏰ Fin: $(date)"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Activer l'environnement: source venv_finetuning/bin/activate"
echo "2. Lancer le fine-tuning: bash train_all_sectors_v2.sh"
echo "3. Monitoring: tensorboard --logdir=./grpo_outputs --port=6006"
echo ""
echo "📚 Documentation:"
echo "- info_Antoine.md : Guide complet"
echo "- README_finetuning.md : Documentation technique"
echo "- VERSIONS_COMPARISON.md : Comparaison des versions"
