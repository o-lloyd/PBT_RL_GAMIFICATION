STRUCTURES_TO_DISPLAY = {

    # PTV56
    'PTV56' : ["PTV 56","PTV56", "ptv 56", "ptv56", 'PTV 57', 'PTV57', 'PTV_57Gy', 'ptv 54', 'CTV 56'], 

    # PTV70
    'PTV70': ["PTV 70","PTV70", "ptv 70", "ptv70", 'PTV_70Gy', 'ptv 66', 'CTV 70'],

    # CTV57
    'CTV57' : ["CTV57", "CTV 3", "CTV56", "ctv 54", "CTV3 56", "CTV 70Gy"],

    # Brain Stem
    'Brain Stem': ['BRAIN_STEM', 'Brainstem_EXPANDED', 'brainstem', 'Brain Stem_EXPANDED'],

    # Spinal Cord
    'Spinal Cord': ['cord extend', 'Cord_EXPANDED', 'cord', 'Spinal Cord_EXPANDED'],

    # Skin / External
    'External' : ["EXTERNAL", "External", "external", "EXT", "Ext", "ext", 'SKIN', 'External ROI 1'],

    # Oral Cavity
    'Oral Cavity': ['oral cavity', 'Oral Avoid', 'oral', 'Oral Cavity'],

    # Avoid
    'Avoid' : ['avoid', 'Avoid', 'tissue avoid']
}

PRIORITIES = {
    'PTV56'        : 4,
    'PTV70'        : 5,
    'Spinal Cord'  : 18,
    'cord extend'  : 17,
    'Oral Cavity'  : 10,
    'Trachea'      : 11,
    'Brain Stem'   : 20,
    'External'     : 1,
    'Avoid'        : 22

}






# STRUCTURES_TO_DISPLAY = [
#     "avoid",          # Region to avoid during treatment
#     "BRAIN_STEM",     # Central part of the brain connecting to the spinal cord
#     "cord extend",    # Extended region of the spinal cord

#     "oral cavity",    # The mouth and its internal structures
#     "PAROTID_LT",     # Left parotid gland, a salivary gland in front of the ear
#     "PAROTID_RT",     # Right parotid gland, a salivary gland in front of the ear
#     "PTV56",          # Planning Target Volume, 56 Gy dose level
#     "PTV70", "PTV 70"         # Planning Target Volume, 70 Gy dose level
#     "SKIN",           # External covering of the body
#     "SPINAL_CORD",    # Major bundle of nerves connecting the brain to the body
#     "SPINL_CRD_PRV",  # Margin around the spinal cord for protection
#     "Trachea",        # Tube connecting the larynx to the lungs
#     "Trans PTV56",    # Transitional or transformed planning volume at 56 Gy
#     "Trans PTV70",    # Transitional or transformed planning volume at 70 Gy
#     "EXTERNAL", "External", "external", "EXT", "Ext", "ext",
#     "CTV56",          # Clinical Target Volume, 56 Gy dose level
#     "CTV70 Sub", "CTV 70 Sub", "CTV70 SUB", "CTV 70 sub", "CTV 70 SUB", "CTV 70Gy", "CTV1 70"
#     "CTV 70", 

#     "GTV","gtv", "Gtv", "gTV",            # Gross Tumor Volume, visible or palpable extent of the tumor
#]


    