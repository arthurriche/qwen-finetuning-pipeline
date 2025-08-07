#!/usr/bin/env python3
"""
Script pour créer les datasets de fine-tuning à partir des datasets mis à jour
Version 8 avec instructions séparées du dataset
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Any
import logging
from itertools import combinations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Charge un dataset JSON"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Les données sont dans structured_data
            if 'structured_data' in data:
                return data['structured_data']
            return data
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de {dataset_path}: {e}")
        return {}

def create_balanced_combinations(items: List[Dict], num_choices: int = 5, max_combinations: int = 5000):
    """
    Créer des combinaisons aléatoires d'items avec position aléatoire du meilleur
    """
    if len(items) < num_choices:
        return []
    
    # Calculer le nombre maximum de combinaisons possibles
    max_possible = len(items) * (len(items) - 1) * (len(items) - 2) * (len(items) - 3) * (len(items) - 4) // 120
    actual_max = min(max_combinations, max_possible)
    
    logger.info(f"   Items disponibles: {len(items)}")
    logger.info(f"   Combinaisons possibles: {max_possible}")
    logger.info(f"   Combinaisons à créer: {actual_max}")
    
    combinations_list = []
    used_combinations = set()
    
    # Créer des combinaisons aléatoires
    attempts = 0
    max_attempts = actual_max * 10  # Limite pour éviter une boucle infinie
    
    while len(combinations_list) < actual_max and attempts < max_attempts:
        attempts += 1
        
        # Sélectionner num_choices items aléatoirement
        selected_items = random.sample(items, num_choices)
        
        # Créer une clé unique pour cette combinaison
        combo_key = tuple(sorted([item.get("keyword", item.get("url", "")) for item in selected_items]))
        
        # Vérifier si cette combinaison n'a pas déjà été utilisée
        if combo_key not in used_combinations:
            used_combinations.add(combo_key)
            
            # Trier par score pour identifier le meilleur
            if 'grpo_score' in selected_items[0]:
                sorted_items = sorted(selected_items, key=lambda x: x.get("grpo_score", 0), reverse=True)
            else:
                sorted_items = sorted(selected_items, key=lambda x: x.get("sft_score", 0))
            
            # Le meilleur est le premier après tri
            best_item = sorted_items[0]
            
            # Mélanger les positions pour plus d'aléatoire
            shuffled_items = sorted_items.copy()
            random.shuffle(shuffled_items)
            
            # Trouver la nouvelle position du meilleur item
            best_index = shuffled_items.index(best_item)
            
            # Extraire tous les scores bruts
            scores = []
            for item in shuffled_items:
                if 'grpo_score' in item:
                    scores.append(str(item.get("grpo_score", 0)))
                else:
                    scores.append(str(item.get("sft_score", 0)))
            
            combinations_list.append({
                "items": shuffled_items,
                "best_index": best_index,
                "best_score": best_item.get("grpo_score", best_item.get("sft_score", 0)),
                "all_scores": scores
            })
    
    logger.info(f"   Combinaisons créées: {len(combinations_list)}")
    logger.info(f"   Tentatives: {attempts}")
    
    return combinations_list

def load_instructions():
    """Charge les instructions depuis le fichier JSON"""
    try:
        with open("instructions_v6.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des instructions: {e}")
        return {}

def get_full_sector_name(domain_name: str) -> str:
    """Convertit le nom de domaine en nom complet du secteur"""
    sector_mapping = {
        "submarine_drones": "submarine drones manufacturers",
        "energy_submetering": "energy submetering",
        "waste_morocco": "hazardous waste treatment in Morocco",
        "its": "intelligent transport systems",
        "smart_water": "smart water services",
        "adhesive": "adhesive tape converters in Belgium",
        "solar_thermal_europe": "solar thermal in europe",
        "full_flight_simulator": "full flight simulators manufacturers"
    }
    return sector_mapping.get(domain_name, domain_name)

def prepare_unified_dataset_v8(structured_data: Dict[str, Any], domain_name: str) -> List[Dict[str, Any]]:
    """
    Préparer le dataset unifié version 8 sans instruction dans le dataset
    """
    logger.info("🔄 Préparation du dataset unifié v8...")
    
    keywords = structured_data.get("keywords_searched", [])
    urls = structured_data.get("urls_with_context", [])
    
    samples = []
    
    # Créer des combinaisons aléatoires pour les keywords (5000 max)
    if len(keywords) >= 5:
        keyword_combinations = create_balanced_combinations(keywords, num_choices=5, max_combinations=5000)
        
        for combo in keyword_combinations:
            items_list = []
            for j, kw in enumerate(combo["items"]):
                items_list.append(f"{j+1}. {kw.get('keyword', '')}")
            
            input_text = "\n".join(items_list)
            
            sample = {
                "input": input_text,
                "output": f"{combo['best_index'] + 1}",
                "score": ",".join(combo["all_scores"]),  # Tous les scores bruts séparés par virgules
                "type": "keyword"
            }
            samples.append(sample)
    
    # Créer des combinaisons aléatoires pour les URLs (20000 max)
    if len(urls) >= 5:
        url_combinations = create_balanced_combinations(urls, num_choices=5, max_combinations=20000)
        
        for combo in url_combinations:
            items_list = []
            for j, url_data in enumerate(combo["items"]):
                url = url_data.get("url", "")
                context = url_data.get("context", "")
                items_list.append(f"{j+1}. {url}")
                if context:
                    items_list[-1] += f" - {context[:100]}..."
            
            input_text = "\n".join(items_list)
            
            sample = {
                "input": input_text,
                "output": f"{combo['best_index'] + 1}",
                "score": ",".join(combo["all_scores"]),  # Tous les scores bruts séparés par virgules
                "type": "url"
            }
            samples.append(sample)
    
    logger.info(f"✅ Préparé {len(samples)} échantillons unifiés v8")
    return samples

def create_finetuning_datasets_for_domain_v8(dataset_path: str, domain_name: str, output_dir: str):
    """Créer le dataset de fine-tuning unifié version 8 pour un domaine spécifique"""
    try:
        logger.info(f"🔄 Création du dataset de fine-tuning unifié v8 pour {domain_name}...")
        
        # Charger le dataset
        structured_data = load_dataset(dataset_path)
        if not structured_data:
            logger.error(f"❌ Pas de données structurées trouvées dans {dataset_path}")
            return None
        
        # Préparer le dataset unifié v8
        samples = prepare_unified_dataset_v8(structured_data, domain_name)
        
        # Créer le dossier de sortie
        domain_output_dir = os.path.join(output_dir, domain_name)
        os.makedirs(domain_output_dir, exist_ok=True)
        
        # Sauvegarder le dataset unifié (seulement les échantillons)
        unified_path = os.path.join(domain_output_dir, "unified_training_data_v8.json")
        
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        # Créer un fichier de métadonnées
        metadata = {
            "domain": domain_name,
            "full_sector_name": get_full_sector_name(domain_name),
            "dataset_path": dataset_path,
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "unified_file": "unified_training_data_v8.json",
            "version": "8.0",
            "format": "unified_choice_based_no_instructions",
            "instruction_type": "external_file_separate",
            "randomization": "high",
            "max_keyword_combinations": 5000,
            "max_url_combinations": 20000,
            "no_mixed_combinations": True,
            "score_format": "raw_scores_comma_separated",
            "response_format": "number_only"
        }
        
        metadata_path = os.path.join(domain_output_dir, "metadata_v8.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Dataset de fine-tuning unifié v8 créé pour {domain_name}:")
        logger.info(f"   - Total: {len(samples)} échantillons")
        logger.info(f"   - Dossier: {domain_output_dir}")
        
        return {
            "domain": domain_name,
            "full_sector_name": get_full_sector_name(domain_name),
            "total_samples": len(samples),
            "output_dir": domain_output_dir,
            "unified_path": unified_path
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du dataset pour {domain_name}: {e}")
        return None

def main():
    """Fonction principale pour créer tous les datasets de fine-tuning unifiés v8"""
    
    # Liste des datasets mis à jour
    datasets = [
        {
            "path": "datasets_finaux/dataset_submarine_drones_updated.json",
            "domain": "submarine_drones"
        },
        {
            "path": "datasets_finaux/dataset_energy_submetering_updated.json",
            "domain": "energy_submetering"
        },
        {
            "path": "datasets_finaux/dataset_waste_morocco_updated.json",
            "domain": "waste_morocco"
        },
        {
            "path": "datasets_finaux/dataset_its_updated.json",
            "domain": "its"
        },
        {
            "path": "datasets_finaux/dataset_smart_water_updated.json",
            "domain": "smart_water"
        },
        {
            "path": "datasets_finaux/dataset_adhesive_updated.json",
            "domain": "adhesive"
        },
        {
            "path": "datasets_finaux/dataset_solar_thermal_europe_updated.json",
            "domain": "solar_thermal_europe"
        },
        {
            "path": "datasets_finaux/dataset_full_flight_simulator_updated.json",
            "domain": "full_flight_simulator"
        }
    ]
    
    # Dossier de sortie pour les datasets de fine-tuning
    output_dir = "finetuning_datasets_v8"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("🚀 Création des datasets de fine-tuning unifiés v8 pour tous les domaines...")
    
    results = []
    successful_domains = []
    failed_domains = []
    
    for dataset_info in datasets:
        dataset_path = dataset_info["path"]
        domain_name = dataset_info["domain"]
        
        if os.path.exists(dataset_path):
            result = create_finetuning_datasets_for_domain_v8(dataset_path, domain_name, output_dir)
            if result:
                results.append(result)
                successful_domains.append(domain_name)
            else:
                failed_domains.append(domain_name)
        else:
            logger.error(f"❌ Dataset non trouvé: {dataset_path}")
            failed_domains.append(domain_name)
    
    # Créer un fichier de résumé
    summary = {
        "created_at": datetime.now().isoformat(),
        "version": "8.0",
        "format": "unified_choice_based_no_instructions",
        "instruction_type": "external_file_separate",
        "randomization": "high",
        "max_keyword_combinations": 5000,
        "max_url_combinations": 20000,
        "no_mixed_combinations": True,
        "score_format": "raw_scores_comma_separated",
        "response_format": "number_only",
        "total_domains": len(datasets),
        "successful_domains": len(successful_domains),
        "failed_domains": len(failed_domains),
        "successful_domains_list": successful_domains,
        "failed_domains_list": failed_domains,
        "results": results
    }
    
    summary_path = os.path.join(output_dir, "summary_v8.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Afficher le résumé
    logger.info(f"\n🎉 Création des datasets de fine-tuning unifiés v8 terminée!")
    logger.info(f"📁 Dossier de sortie: {output_dir}")
    logger.info(f"✅ Domaines traités avec succès ({len(successful_domains)}):")
    for domain in successful_domains:
        logger.info(f"   - {domain}")
    
    if failed_domains:
        logger.info(f"❌ Domaines en échec ({len(failed_domains)}):")
        for domain in failed_domains:
            logger.info(f"   - {domain}")
    
    # Statistiques globales
    total_samples = sum(r["total_samples"] for r in results)
    
    logger.info(f"\n📊 Statistiques globales:")
    logger.info(f"   - Total échantillons: {total_samples}")
    logger.info(f"   - Fichier de résumé: {summary_path}")

if __name__ == "__main__":
    main() 