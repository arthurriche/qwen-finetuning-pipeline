#!/bin/bash
# Script de lancement pour le fine-tuning de tous les secteurs
# Usage: bash train_all_sectors.sh

set -e  # Arrêter en cas d'erreur

# Configuration
sectors=("submarine_drones" "energy_submetering" "waste_morocco" "its" "smart_water" "adhesive" "solar_thermal_europe" "full_flight_simulator")

echo "🚀 Démarrage du fine-tuning v2 avec vraies instructions pour tous les secteurs..."
echo "📊 Secteurs à traiter: ${sectors[*]}"
echo "⏰ Début: $(date)"

# Créer les dossiers de sortie
mkdir -p grpo_outputs
mkdir -p sft_outputs

# Fonction pour modifier le secteur dans un script Python
modify_sector() {
    local script_file=$1
    local sector=$2
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/sector = \".*\"/sector = \"$sector\"/" "$script_file"
    else
        # Linux
        sed -i "s/sector = \".*\"/sector = \"$sector\"/" "$script_file"
    fi
}

# Traitement de chaque secteur
for sector in "${sectors[@]}"; do
    echo ""
    echo "🔄 Traitement du secteur: $sector"
    echo "⏰ $(date)"
    
    # GRPO Training
    echo "📈 Lancement du fine-tuning GRPO pour $sector..."
    modify_sector "grpo_finetuning_v2.py" "$sector"
    python grpo_finetuning_v2.py
    
    if [ $? -eq 0 ]; then
        echo "✅ GRPO training v2 terminé avec succès pour $sector"
    else
        echo "❌ Erreur lors du GRPO training v2 pour $sector"
        exit 1
    fi
    
    # SFT Training
    echo "📈 Lancement du fine-tuning SFT pour $sector..."
    modify_sector "sft_finetuning_v2.py" "$sector"
    python sft_finetuning_v2.py
    
    if [ $? -eq 0 ]; then
        echo "✅ SFT training v2 terminé avec succès pour $sector"
    else
        echo "❌ Erreur lors du SFT training v2 pour $sector"
        exit 1
    fi
    
    echo "✅ Secteur $sector terminé avec succès"
done

echo ""
echo "🎉 Fine-tuning v2 terminé pour tous les secteurs!"
echo "⏰ Fin: $(date)"
echo ""
echo "📁 Résultats disponibles dans:"
echo "   - GRPO: ./grpo_outputs/"
echo "   - SFT: ./sft_outputs/"
echo ""
echo "📊 Monitoring TensorBoard:"
echo "   - GRPO: tensorboard --logdir=./grpo_outputs --port=6006"
echo "   - SFT: tensorboard --logdir=./sft_outputs --port=6007" 