# Data sources
database(
    thermoLibraries = ['primaryThermoLibrary'],
    reactionLibraries = [],
    seedMechanisms = [],
    kineticsDepositories = ['training'],
    kineticsFamilies = 'default',
    kineticsEstimator = 'rate rules',
)

# List of species
species(
    label='EA',
    reactive=True,
    structure=SMILES("CC(=O)OC"),
)

species(
    label='Ar',
    reactive=False,
    structure=SMILES("[Ar]"),
)

# Reaction systems
simpleReactor(
    temperature=(1000,'K'),
    pressure=(0.1,'bar'),
    initialMoleFractions={
        "EA": 0.1,
        "Ar":0.9
    },
    terminationConversion={
        'EA': 0.99,
    },
    terminationTime=(10,'s'),
)

simulator(
    atol=1e-16,
    rtol=1e-8,
)

model(
    toleranceMoveToCore=0.5,
    toleranceInterruptSimulation=1e8,
    toleranceKeepInEdge=0.05,
    maximumEdgeSpecies=200000,
    minCoreSizeForPrune=50,
    minSpeciesExistIterationsForPrune=2,
)

options(
    units='si',
    generateOutputHTML=True,
    generatePlots=False,
    saveEdgeSpecies=True,
    saveSimulationProfiles=True,
)
