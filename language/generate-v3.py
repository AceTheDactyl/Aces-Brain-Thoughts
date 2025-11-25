#!/usr/bin/env python3
"""
Generate v3 Language Pack with APL Tier-3 Chemical Integration
"""
import json
from datetime import datetime, timezone

# Load current language pack
with open('latest.json', 'r') as f:
    pack = json.load(f)

# APL Chemical mappings based on neurotransmitter systems and neural function
NEURAL_CHEMICAL_MAP = {
    "I": {  # Somatosensory Cortex
        "primary_nt": "glutamate",
        "apl_tokens": ["e:U(ionize)TRUE@3", "Φ:M(bond)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "bond", "charge"]
    },
    "II": {  # ACC
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(redox)TRUE@3", "Φ:C(complex)TRUE@3", "e:D(reduce)UNTRUE@3"],
        "operators": ["redox", "complex", "reduce"]
    },
    "III": {  # Thalamus
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:Mod(fold)TRUE@3", "π:M(crystallize)TRUE@3"],
        "operators": ["ionize", "fold", "crystallize"]
    },
    "IV": {  # Motor Cortex
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:U(bond)TRUE@3", "e:E(excite)TRUE@3", "Φ:C(polymerize)TRUE@3"],
        "operators": ["bond", "excite", "polymerize"]
    },
    "V": {  # Broca's Area
        "primary_nt": "dopamine",
        "apl_tokens": ["e:U(excite)TRUE@3", "Φ:E(polymerize)TRUE@3", "e:C(charge)TRUE@3"],
        "operators": ["excite", "polymerize", "charge"]
    },
    "VI": {  # Mirror Neurons
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(resonate)TRUE@3", "Φ:C(complex)TRUE@3", "e:Mod(fold)TRUE@3"],
        "operators": ["resonate", "complex", "fold"]
    },
    "VII": {  # Amygdala
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:U(oxidize)TRUE@3", "π:U(charge)TRUE@3"],
        "operators": ["excite", "oxidize", "charge"]
    },
    "VIII": {  # PFC
        "primary_nt": "dopamine",
        "apl_tokens": ["e:Mod(catalyze)TRUE@3", "Φ:M(complex)TRUE@3", "e:C(redox)TRUE@3"],
        "operators": ["catalyze", "complex", "redox"]
    },
    "IX": {  # Parietal Eye Field
        "primary_nt": "acetylcholine",
        "apl_tokens": ["e:U(charge)TRUE@3", "Φ:C(bond)TRUE@3", "e:Mod(ionize)TRUE@3"],
        "operators": ["charge", "bond", "ionize"]
    },
    "X": {  # Subiculum
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:M(crystallize)TRUE@3", "e:E(bond)TRUE@3", "π:C(fold)TRUE@3"],
        "operators": ["crystallize", "bond", "fold"]
    },
    "XI": {  # Pineal
        "primary_nt": "melatonin",
        "apl_tokens": ["π:D(relax)TRUE@3", "e:M(reduce)TRUE@3", "Φ:Mod(fold)TRUE@3"],
        "operators": ["relax", "reduce", "fold"]
    },
    "XII": {  # MTG
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:C(polymerize)TRUE@3", "e:M(complex)TRUE@3", "Φ:E(bond)TRUE@3"],
        "operators": ["polymerize", "complex", "bond"]
    },
    "XIII": {  # Fastigial-Vestibular
        "primary_nt": "glutamate",
        "apl_tokens": ["π:M(crystallize)TRUE@3", "Φ:C(fold)TRUE@3", "e:M(relax)TRUE@3"],
        "operators": ["crystallize", "fold", "relax"]
    },
    "XIV": {  # Posterior Thalamic
        "primary_nt": "glutamate",
        "apl_tokens": ["e:E(ionize)TRUE@3", "Φ:C(complex)TRUE@3", "π:M(precipitate)TRUE@3"],
        "operators": ["ionize", "complex", "precipitate"]
    },
    "XV": {  # Cerebellar Uvula
        "primary_nt": "GABA",
        "apl_tokens": ["π:D(relax)TRUE@3", "Φ:M(crystallize)TRUE@3", "e:D(reduce)TRUE@3"],
        "operators": ["relax", "crystallize", "reduce"]
    },
    "XVI": {  # AIPS
        "primary_nt": "dopamine",
        "apl_tokens": ["Φ:U(polymerize)TRUE@3", "e:E(excite)TRUE@3", "Φ:C(bond)TRUE@3"],
        "operators": ["polymerize", "excite", "bond"]
    },
    "XVII": {  # Ventrolateral Thalamus
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:M(fold)TRUE@3", "e:Mod(resonate)TRUE@3"],
        "operators": ["ionize", "fold", "resonate"]
    },
    "XVIII": {  # Thalamus Extended
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "π:M(crystallize)TRUE@3", "Φ:Mod(fold)TRUE@3"],
        "operators": ["ionize", "crystallize", "fold"]
    },
    "XIX": {  # Motor Extended
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:U(bond)TRUE@3", "e:E(excite)TRUE@3", "Φ:M(polymerize)TRUE@3"],
        "operators": ["bond", "excite", "polymerize"]
    },
    "XX": {  # Broca's Extended
        "primary_nt": "dopamine",
        "apl_tokens": ["e:U(excite)TRUE@3", "Φ:E(polymerize)TRUE@3", "e:M(resonate)TRUE@3"],
        "operators": ["excite", "polymerize", "resonate"]
    },
    "XXI": {  # Mirror Extended
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(resonate)TRUE@3", "Φ:C(complex)TRUE@3", "e:E(fold)TRUE@3"],
        "operators": ["resonate", "complex", "fold"]
    },
    "XXII": {  # Amygdala Extended
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:M(oxidize)TRUE@3", "π:C(charge)TRUE@3"],
        "operators": ["excite", "oxidize", "charge"]
    },
    "XXIII": {  # PFC Extended
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(catalyze)TRUE@3", "Φ:M(complex)TRUE@3", "e:D(relax)TRUE@3"],
        "operators": ["catalyze", "complex", "relax"]
    },
    "XXIV": {  # OFC
        "primary_nt": "serotonin",
        "apl_tokens": ["e:M(resonate)TRUE@3", "Φ:C(complex)TRUE@3", "π:M(crystallize)TRUE@3"],
        "operators": ["resonate", "complex", "crystallize"]
    },
    "XXV": {  # Cingulate Gyrus
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(redox)TRUE@3", "Φ:C(complex)TRUE@3", "e:Mod(catalyze)TRUE@3"],
        "operators": ["redox", "complex", "catalyze"]
    },
    "XXVI": {  # Ventral Striatum
        "primary_nt": "dopamine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:U(charge)TRUE@3", "Φ:U(ionize)TRUE@3"],
        "operators": ["excite", "charge", "ionize"]
    },
    "XXVII": {  # Claustrum
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(fold)TRUE@3", "e:C(complex)TRUE@3", "Φ:Mod(resonate)TRUE@3"],
        "operators": ["fold", "complex", "resonate"]
    },
    "XXVIII": {  # DMN
        "primary_nt": "acetylcholine",
        "apl_tokens": ["e:M(resonate)TRUE@3", "Φ:M(fold)TRUE@3", "π:M(crystallize)TRUE@3"],
        "operators": ["resonate", "fold", "crystallize"]
    },
    "XXIX": {  # Pineal revisit
        "primary_nt": "melatonin",
        "apl_tokens": ["π:D(relax)TRUE@3", "e:D(reduce)TRUE@3", "Φ:M(fold)TRUE@3"],
        "operators": ["relax", "reduce", "fold"]
    },
    "XXX": {  # Corpus Callosum
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:C(bond)TRUE@3", "e:C(ionize)TRUE@3", "Φ:M(resonate)TRUE@3"],
        "operators": ["bond", "ionize", "resonate"]
    },
    "XXXI": {  # Locus Coeruleus
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:U(oxidize)TRUE@3", "Φ:U(redox)TRUE@3"],
        "operators": ["excite", "oxidize", "redox"]
    },
    "XXXII": {  # PAG
        "primary_nt": "GABA",
        "apl_tokens": ["π:D(reduce)TRUE@3", "e:D(relax)TRUE@3", "Φ:D(unbond)TRUE@3"],
        "operators": ["reduce", "relax", "unbond"]
    },
    "XXXIII": {  # ATP
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:E(polymerize)TRUE@3", "e:M(complex)TRUE@3", "Φ:C(bond)TRUE@3"],
        "operators": ["polymerize", "complex", "bond"]
    },
    "XXXIV": {  # vmPFC
        "primary_nt": "serotonin",
        "apl_tokens": ["e:M(redox)TRUE@3", "Φ:M(fold)TRUE@3", "e:D(relax)TRUE@3"],
        "operators": ["redox", "fold", "relax"]
    },
    "XXXV": {  # Dorsal Raphe
        "primary_nt": "serotonin",
        "apl_tokens": ["e:M(relax)TRUE@3", "Φ:Mod(fold)TRUE@3", "e:D(reduce)TRUE@3"],
        "operators": ["relax", "fold", "reduce"]
    },
    "XXXVI": {  # Superior Colliculus
        "primary_nt": "glutamate",
        "apl_tokens": ["e:U(ionize)TRUE@3", "Φ:C(bond)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "bond", "charge"]
    },
    "XXXVII": {  # Anterior Insula
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(resonate)TRUE@3", "Φ:E(fold)TRUE@3", "e:C(complex)TRUE@3"],
        "operators": ["resonate", "fold", "complex"]
    },
    "XXXVIII": {  # Lateral Habenula
        "primary_nt": "glutamate",
        "apl_tokens": ["e:D(reduce)UNTRUE@3", "Φ:D(unbond)TRUE@3", "π:D(dissolve)TRUE@3"],
        "operators": ["reduce", "unbond", "dissolve"]
    },
    "XXXIX": {  # Precuneus
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(fold)TRUE@3", "e:M(resonate)TRUE@3", "π:M(crystallize)TRUE@3"],
        "operators": ["fold", "resonate", "crystallize"]
    },
    "XL": {  # Cerebellar Cognitive
        "primary_nt": "GABA",
        "apl_tokens": ["e:Mod(catalyze)TRUE@3", "Φ:M(fold)TRUE@3", "π:M(crystallize)TRUE@3"],
        "operators": ["catalyze", "fold", "crystallize"]
    },
    "XLI": {  # Basolateral Amygdala
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:M(excite)TRUE@3", "Φ:E(crystallize)TRUE@3", "e:C(complex)TRUE@3"],
        "operators": ["excite", "crystallize", "complex"]
    },
    "XLII": {  # Pulvinar
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:Mod(fold)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "fold", "charge"]
    },
    "XLIII": {  # TPJ
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(complex)TRUE@3", "e:M(resonate)TRUE@3", "Φ:C(fold)TRUE@3"],
        "operators": ["complex", "resonate", "fold"]
    },
    "XLIV": {  # Medial Septum
        "primary_nt": "acetylcholine",
        "apl_tokens": ["e:Mod(catalyze)TRUE@3", "Φ:C(bond)TRUE@3", "e:M(resonate)TRUE@3"],
        "operators": ["catalyze", "bond", "resonate"]
    },
    "XLV": {  # Subgenual Cingulate
        "primary_nt": "serotonin",
        "apl_tokens": ["e:D(relax)TRUE@3", "Φ:D(dissolve)UNTRUE@3", "π:D(reduce)TRUE@3"],
        "operators": ["relax", "dissolve", "reduce"]
    },
    "XLVI": {  # VTA
        "primary_nt": "dopamine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:U(charge)TRUE@3", "Φ:U(ionize)TRUE@3"],
        "operators": ["excite", "charge", "ionize"]
    },
    "XLVII": {  # Entorhinal
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(crystallize)TRUE@3", "e:C(bond)TRUE@3", "Φ:E(fold)TRUE@3"],
        "operators": ["crystallize", "bond", "fold"]
    },
    "XLVIII": {  # Supramarginal
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:M(complex)TRUE@3", "e:C(resonate)TRUE@3", "Φ:E(fold)TRUE@3"],
        "operators": ["complex", "resonate", "fold"]
    },
    "XLIX": {  # NAcc
        "primary_nt": "dopamine",
        "apl_tokens": ["e:E(reduce)TRUE@3", "Φ:C(complex)TRUE@3", "e:M(resonate)TRUE@3"],
        "operators": ["reduce", "complex", "resonate"]
    },
    "L": {  # Cerebral Aqueduct
        "primary_nt": "GABA",
        "apl_tokens": ["π:D(dissolve)TRUE@3", "e:D(reduce)TRUE@3", "Φ:M(unbond)TRUE@3"],
        "operators": ["dissolve", "reduce", "unbond"]
    },
    "LI": {  # Anterior Thalamic
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:M(fold)TRUE@3", "π:C(crystallize)TRUE@3"],
        "operators": ["ionize", "fold", "crystallize"]
    },
    "LII": {  # Parafascicular
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(charge)TRUE@3", "Φ:Mod(bond)TRUE@3", "e:E(ionize)TRUE@3"],
        "operators": ["charge", "bond", "ionize"]
    },
    "LIII": {  # Inferior Colliculus
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:M(resonate)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "resonate", "charge"]
    },
    "LIV": {  # Perirhinal
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(complex)TRUE@3", "e:E(bond)TRUE@3", "Φ:C(fold)TRUE@3"],
        "operators": ["complex", "bond", "fold"]
    },
    "LV": {  # Vermis
        "primary_nt": "GABA",
        "apl_tokens": ["π:M(crystallize)TRUE@3", "Φ:M(fold)TRUE@3", "e:D(relax)TRUE@3"],
        "operators": ["crystallize", "fold", "relax"]
    },
    "LVI": {  # Anterior Insular-Operculum
        "primary_nt": "dopamine",
        "apl_tokens": ["e:E(resonate)TRUE@3", "Φ:C(complex)TRUE@3", "e:M(fold)TRUE@3"],
        "operators": ["resonate", "complex", "fold"]
    },
    "LVII": {  # Paraventricular
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:U(excite)TRUE@3", "π:U(charge)TRUE@3", "e:M(oxidize)TRUE@3"],
        "operators": ["excite", "charge", "oxidize"]
    },
    "LVIII": {  # Lateral OFC
        "primary_nt": "serotonin",
        "apl_tokens": ["e:M(redox)TRUE@3", "Φ:C(complex)TRUE@3", "e:D(reduce)TRUE@3"],
        "operators": ["redox", "complex", "reduce"]
    },
    "LIX": {  # Midcingulate
        "primary_nt": "dopamine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:Mod(catalyze)TRUE@3", "Φ:E(bond)TRUE@3"],
        "operators": ["excite", "catalyze", "bond"]
    },
    "LX": {  # Calcarine
        "primary_nt": "glutamate",
        "apl_tokens": ["e:U(ionize)TRUE@3", "Φ:E(charge)TRUE@3", "e:C(excite)TRUE@3"],
        "operators": ["ionize", "charge", "excite"]
    },
    "LXI": {  # Rostral PFC
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(catalyze)TRUE@3", "Φ:M(complex)TRUE@3", "e:Mod(redox)TRUE@3"],
        "operators": ["catalyze", "complex", "redox"]
    },
    "LXII": {  # MLR
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:U(bond)TRUE@3", "e:U(excite)TRUE@3", "e:C(charge)TRUE@3"],
        "operators": ["bond", "excite", "charge"]
    },
    "LXIII": {  # Anterior Temporal Sulcus
        "primary_nt": "glutamate",
        "apl_tokens": ["Φ:E(polymerize)TRUE@3", "e:M(complex)TRUE@3", "Φ:C(bond)TRUE@3"],
        "operators": ["polymerize", "complex", "bond"]
    },
    "LXIV": {  # Lateral Septum
        "primary_nt": "GABA",
        "apl_tokens": ["e:D(relax)TRUE@3", "π:D(reduce)TRUE@3", "Φ:M(fold)TRUE@3"],
        "operators": ["relax", "reduce", "fold"]
    },
    "LXV": {  # Cerebellar Tonsil
        "primary_nt": "GABA",
        "apl_tokens": ["e:D(relax)TRUE@3", "π:M(crystallize)TRUE@3", "Φ:D(reduce)TRUE@3"],
        "operators": ["relax", "crystallize", "reduce"]
    },
    "LXVI": {  # Pontine Reticular
        "primary_nt": "acetylcholine",
        "apl_tokens": ["e:U(excite)TRUE@3", "Φ:U(bond)TRUE@3", "e:Mod(catalyze)TRUE@3"],
        "operators": ["excite", "bond", "catalyze"]
    },
    "LXVII": {  # Insular-Opercular Speech
        "primary_nt": "dopamine",
        "apl_tokens": ["e:E(resonate)TRUE@3", "Φ:E(polymerize)TRUE@3", "e:C(charge)TRUE@3"],
        "operators": ["resonate", "polymerize", "charge"]
    },
    "LXVIII": {  # Amygdala Central
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:U(oxidize)TRUE@3", "Φ:D(unbond)TRUE@3"],
        "operators": ["excite", "oxidize", "unbond"]
    },
    "LXIX": {  # TRN
        "primary_nt": "GABA",
        "apl_tokens": ["e:Mod(ionize)TRUE@3", "Φ:C(fold)TRUE@3", "π:M(reduce)TRUE@3"],
        "operators": ["ionize", "fold", "reduce"]
    },
    "LXX": {  # Cuneus
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:M(fold)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "fold", "charge"]
    },
    "LXXI": {  # VMH
        "primary_nt": "GABA",
        "apl_tokens": ["π:M(crystallize)TRUE@3", "e:M(relax)TRUE@3", "Φ:M(fold)TRUE@3"],
        "operators": ["crystallize", "relax", "fold"]
    },
    "LXXII": {  # Periventricular Gray
        "primary_nt": "GABA",
        "apl_tokens": ["π:D(reduce)TRUE@3", "e:D(relax)TRUE@3", "Φ:M(unbond)TRUE@3"],
        "operators": ["reduce", "relax", "unbond"]
    },
    "LXXIII": {  # Frontal Operculum
        "primary_nt": "dopamine",
        "apl_tokens": ["e:E(resonate)TRUE@3", "Φ:E(polymerize)TRUE@3", "e:C(charge)TRUE@3"],
        "operators": ["resonate", "polymerize", "charge"]
    },
    "LXXIV": {  # Nodulus
        "primary_nt": "GABA",
        "apl_tokens": ["π:M(crystallize)TRUE@3", "Φ:M(fold)TRUE@3", "e:D(relax)TRUE@3"],
        "operators": ["crystallize", "fold", "relax"]
    },
    "LXXV": {  # Substantia Nigra
        "primary_nt": "dopamine",
        "apl_tokens": ["e:C(redox)TRUE@3", "Φ:M(catalyze)TRUE@3", "e:E(oxidize)TRUE@3"],
        "operators": ["redox", "catalyze", "oxidize"]
    },
    "LXXVI": {  # V4
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:E(fold)TRUE@3", "e:M(resonate)TRUE@3"],
        "operators": ["ionize", "fold", "resonate"]
    },
    "LXXVII": {  # Lingual
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:M(complex)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "complex", "charge"]
    },
    "LXXVIII": {  # mPFC
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(redox)TRUE@3", "Φ:M(fold)TRUE@3", "e:Mod(catalyze)TRUE@3"],
        "operators": ["redox", "fold", "catalyze"]
    },
    "LXXIX": {  # dLPFC
        "primary_nt": "dopamine",
        "apl_tokens": ["e:E(redox)TRUE@3", "Φ:C(complex)TRUE@3", "e:Mod(catalyze)TRUE@3"],
        "operators": ["redox", "complex", "catalyze"]
    },
    "LXXX": {  # IPL
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(complex)TRUE@3", "e:M(resonate)TRUE@3", "Φ:E(fold)PARADOX@3"],
        "operators": ["complex", "resonate", "fold"]
    },
    "LXXXI": {  # ACC revisit
        "primary_nt": "dopamine",
        "apl_tokens": ["e:M(redox)TRUE@3", "Φ:C(complex)TRUE@3", "e:Mod(catalyze)TRUE@3"],
        "operators": ["redox", "complex", "catalyze"]
    },
    "LXXXII": {  # Anterior Hippocampus
        "primary_nt": "acetylcholine",
        "apl_tokens": ["Φ:M(crystallize)TRUE@3", "e:C(bond)TRUE@3", "Φ:E(fold)TRUE@3"],
        "operators": ["crystallize", "bond", "fold"]
    },
    "LXXXIII": {  # Crus I/II
        "primary_nt": "GABA",
        "apl_tokens": ["e:Mod(catalyze)TRUE@3", "Φ:M(fold)TRUE@3", "π:M(crystallize)TRUE@3"],
        "operators": ["catalyze", "fold", "crystallize"]
    },
    "LXXXIV": {  # Basal Forebrain
        "primary_nt": "acetylcholine",
        "apl_tokens": ["e:Mod(catalyze)TRUE@3", "e:C(charge)TRUE@3", "Φ:M(ionize)TRUE@3"],
        "operators": ["catalyze", "charge", "ionize"]
    },
    "LXXXV": {  # Reticular Formation
        "primary_nt": "norepinephrine",
        "apl_tokens": ["e:U(excite)TRUE@3", "e:Mod(catalyze)TRUE@3", "Φ:U(bond)TRUE@3"],
        "operators": ["excite", "catalyze", "bond"]
    },
    "LXXXVI": {  # DVC
        "primary_nt": "GABA",
        "apl_tokens": ["π:D(reduce)TRUE@3", "e:D(relax)TRUE@3", "Φ:D(dissolve)TRUE@3"],
        "operators": ["reduce", "relax", "dissolve"]
    },
    "LXXXVII": {  # Cranial Nerves
        "primary_nt": "acetylcholine",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:E(bond)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "bond", "charge"]
    },
    "LXXXVIII": {  # Spinal Relays
        "primary_nt": "glutamate",
        "apl_tokens": ["e:C(ionize)TRUE@3", "Φ:C(bond)TRUE@3", "e:E(charge)TRUE@3"],
        "operators": ["ionize", "bond", "charge"]
    },
    "LXXXIX": {  # Globus Pallidus
        "primary_nt": "GABA",
        "apl_tokens": ["π:M(reduce)TRUE@3", "e:C(relax)TRUE@3", "Φ:Mod(unbond)TRUE@3"],
        "operators": ["reduce", "relax", "unbond"]
    },
    "XC": {  # Lateral Hypothalamus
        "primary_nt": "orexin",
        "apl_tokens": ["e:U(excite)TRUE@3", "π:U(charge)TRUE@3", "e:M(oxidize)TRUE@3"],
        "operators": ["excite", "charge", "oxidize"]
    }
}

# Phase-based default mappings for tokens not explicitly mapped
PHASE_DEFAULTS = {
    "Ignition": {
        "apl_tokens": ["e:U(excite)TRUE@3", "e:U(ionize)TRUE@3", "e:U(charge)TRUE@3"],
        "operators": ["excite", "ionize", "charge"]
    },
    "Empowerment": {
        "apl_tokens": ["Φ:U(bond)TRUE@3", "Φ:U(polymerize)TRUE@3", "e:U(catalyze)TRUE@3"],
        "operators": ["bond", "polymerize", "catalyze"]
    },
    "Resonance": {
        "apl_tokens": ["e:M(resonate)TRUE@3", "Φ:M(complex)TRUE@3", "e:M(redox)TRUE@3"],
        "operators": ["resonate", "complex", "redox"]
    },
    "Mania": {
        "apl_tokens": ["e:E(excite)TRUE@3", "e:E(oxidize)TRUE@3", "e:E(charge)PARADOX@3"],
        "operators": ["excite", "oxidize", "charge"]
    },
    "Nirvana": {
        "apl_tokens": ["π:M(crystallize)TRUE@3", "e:M(relax)TRUE@3", "Φ:M(fold)TRUE@3"],
        "operators": ["crystallize", "relax", "fold"]
    },
    "Transmission": {
        "apl_tokens": ["e:C(ionize)TRUE@3", "e:C(charge)TRUE@3", "Φ:C(bond)TRUE@3"],
        "operators": ["ionize", "charge", "bond"]
    },
    "Pause": {
        "apl_tokens": ["π:D(relax)UNTRUE@3", "e:D(reduce)UNTRUE@3", "Φ:D(unbond)UNTRUE@3"],
        "operators": ["relax", "reduce", "unbond"]
    }
}

def get_primary_phase(phase_str):
    """Extract primary phase from compound phase strings like 'Ignition → Empowerment'"""
    if '→' in phase_str:
        return phase_str.split('→')[0].strip()
    return phase_str.strip()

def get_apl_mapping(token_id, phase):
    """Get APL mapping for a token, using explicit mapping or phase default"""
    if token_id in NEURAL_CHEMICAL_MAP:
        mapping = NEURAL_CHEMICAL_MAP[token_id]
        return {
            "apl_tier": 3,
            "domain": "chemical",
            "primary_neurotransmitter": mapping["primary_nt"],
            "apl_tokens": mapping["apl_tokens"],
            "dominant_operators": mapping["operators"]
        }

    # Fall back to phase-based default
    primary_phase = get_primary_phase(phase)
    if primary_phase in PHASE_DEFAULTS:
        defaults = PHASE_DEFAULTS[primary_phase]
        return {
            "apl_tier": 3,
            "domain": "chemical",
            "primary_neurotransmitter": "inferred",
            "apl_tokens": defaults["apl_tokens"],
            "dominant_operators": defaults["operators"]
        }

    return {
        "apl_tier": 3,
        "domain": "chemical",
        "primary_neurotransmitter": "unknown",
        "apl_tokens": [],
        "dominant_operators": []
    }

# Update pack meta
pack["meta"]["version"] = "v3"
pack["meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()
pack["meta"]["sources"].append("apl-tier3-chemical.json")
pack["meta"]["sources"].append("wumbo-apl-chemical-bridge.json")

# Add APL integration section to theory
pack["meta"]["theory"]["apl_integration"] = {
    "tier": 3,
    "domain": "chemical",
    "token_format": "Field:Machine(Operator)TruthState@Tier",
    "fields": ["Φ", "e", "π"],
    "machines": ["U", "D", "M", "E", "C", "Mod"],
    "chemical_operators": [
        "bond", "unbond", "crystallize", "dissolve", "ionize", "redox",
        "catalyze", "polymerize", "depolymerize", "fold", "charge", "excite",
        "relax", "oxidize", "reduce", "precipitate", "complex", "resonate"
    ],
    "truth_states": ["TRUE", "UNTRUE", "PARADOX"],
    "total_chemical_tokens": 972
}

# Update each token with APL mapping
for token in pack["tokens"]:
    token_id = token["id"]
    phase = token.get("phase", "")

    # Get APL mapping
    apl_mapping = get_apl_mapping(token_id, phase)

    # Update wumbo_roles with APL integration
    token["wumbo_roles"] = [apl_mapping]

# Write v3 pack
with open('v3/pack.json', 'w') as f:
    json.dump(pack, f, indent=2)

# Also update latest.json
with open('latest.json', 'w') as f:
    json.dump(pack, f, indent=2)

print(f"Generated v3 language pack with {len(pack['tokens'])} tokens")
print(f"Each token now has APL Tier-3 Chemical mappings in wumbo_roles")
