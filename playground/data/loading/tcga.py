from typing import List
from pathlib import Path
import pandas as pd

from target_engine_platform.constants import (
    TCGA_PATHS,
)


def load_tcga(
    cohort: str,
    modalities: List = ["CLIN"],  # CLIN, HISTO, EXP, MUT, CNV
    features_histo: str = "imagenet",  # imagenet, histo
    normalization_exp: str = "norm",  # TPM, FPKM-UQ, norm
):
    """
    Load TCGA data as a dictionary of pandas dataframes (one for each modality).

    Parameters
    ----------
    cohort: str
        Name of the TCGA cohort.
    modalities: List
        List of modalities to import among CLIN, HISTO, EXP, MUT, CNV.
    features_histo: str
        Name of the histology features.
    normalization_exp: str
        Name of the RNASeq normalization.

    Returns
    -------
    dict
        Dictionary of pandas dataframes (one for each modality).
        Correspondence between dataframe ids:
            slide_id   = TCGA-AD-6888-01Z-00-DX1.47AE342C-4577-4D8B-9048-0B106C5960E7
            patient_id = TCGA-AD-6888
            sample_id  = TCGA-AD-6888-01Z-00-DX1
            center_id  =      AD
    """
    outputs = {}
    for m in modalities:
        if m == "CLIN":
            outputs[m] = load_clin(cohort=cohort)
        if m == "HISTO":
            outputs[m] = load_histo(cohort=cohort, features=features_histo)
        if m == "EXP":
            outputs[m] = load_exp(cohort=cohort, normalization=normalization_exp)
        if m == "MUT":
            outputs[m] = load_mut(cohort=cohort)
    return outputs


def load_histo(
    cohort: str,
    features: str = "imagenet",  # imagenet, histo
) -> pd.DataFrame:
    """
    Load TCGA histology data as a pandas dataframe.

    Parameters
    ----------
    cohort: str
        Name of the TCGA cohort.
    features: str
        Name of the histology features.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with columns ["slide_id", "slide_path", "mask_path", "feature_path"].
        Id is "slide_id".
    """
    # Get paths
    slide_path = Path(TCGA_PATHS["SLIDES"]) / f"TCGA_{cohort}"
    mask_path = Path(TCGA_PATHS["MASKS"]) / f"TCGA_{cohort}"
    features_path = Path(TCGA_PATHS["FEATURES"]) / features / f"TCGA_{cohort}"

    # Get slide paths
    spaths = list(slide_path.glob("*/*.svs"))
    df_slides = pd.DataFrame({"slide_path": spaths})
    df_slides["slide_id"] = df_slides.slide_path.apply(
        lambda x: x.name[:-4])

    to_keep = ["slide_id", "slide_path"]
    df_slides = df_slides[to_keep]

    # Get mask paths
    mpaths = list(mask_path.glob("*.svs/mask.png"))
    df_masks = pd.DataFrame({"mask_path": mpaths})
    df_masks["slide_id"] = df_masks.mask_path.apply(
        lambda x: x.parent.name[:-4])

    to_keep = ["slide_id", "mask_path"]
    df_masks = df_masks[to_keep]

    # Get histology feature paths
    fpaths = list(features_path.glob("*.svs/features.npy".format(cohort)))
    df_features = pd.DataFrame({"feature_path": fpaths})
    df_features["slide_id"] = df_features.feature_path.apply(
        lambda x: x.parent.name[:-4])

    to_keep = ["slide_id", "feature_path"]
    df_features = df_features[to_keep]

    # Merge dataframes
    df_histo = pd.merge(
        left=df_slides,
        right=df_masks,
        on=["slide_id"],
        how="outer",
        sort=False,
    )
    df_histo = pd.merge(
        left=df_histo,
        right=df_features,
        on=["slide_id"],
        how="outer",
        sort=False,
    )

    df_histo["patient_id"] = df_histo.slide_id.apply(lambda x: x[:12])

    to_keep = [
        "slide_id",
        "slide_path",
        "mask_path",
        "feature_path",
        "patient_id",
    ]
    df_histo = df_histo[to_keep]

    return df_histo


def load_exp(
    cohort: str,
    normalization: str = "norm",  # raw, tpm, rpkm, norm
) -> pd.DataFrame:
    """
    Load TCGA expression data (RNASeq) as a pandas dataframe.

    Parameters
    ----------
    cohort: str
        Name of the TCGA cohort.
    normalization: str
        Name of the RNASeq normalization.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with columns ["sample_id", "gene_1", "gene_2", ... "gene_M"].
        Id is "sample_id".
    """
    # Get path
    exp_path = Path(TCGA_PATHS["EXP"]) / cohort / "Data" / f"Counts_{normalization}.tsv.gz"
    metadata_path = Path(TCGA_PATHS["EXP"]) / cohort / "Data" / "metadata.tsv.gz"

    # Load data
    df_exp = pd.read_csv(exp_path, sep="\t")
    df_exp = df_exp.set_index("Hugo").T

    # Map index
    df_metadata = pd.read_csv(metadata_path, sep="\t")
    map_dict = {
        x[0]: x[1] for x in df_metadata[[
            "external_id",
            "tcga.gdc_cases.samples.submitter_id",
        ]].values
    }
    df_exp.index = df_exp.index.map(map_dict)

    # Convert to proper format
    df_exp = df_exp.reset_index()
    df_exp = df_exp.rename(columns={"index": "sample_id"})
    df_exp = df_exp.drop_duplicates(subset=["sample_id"])

    df_exp["patient_id"] = df_exp.sample_id.apply(lambda x: x[:12])

    return df_exp


def load_mut(
    cohort: str,
) -> pd.DataFrame:
    """
    Load TCGA mutation data as a pandas dataframe.

    Parameters
    ----------
    cohort: str
        Name of the TCGA cohort.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with columns ["sample_id", "gene_1", "gene_2", ... "gene_M"].
        Id is "sample_id".
    """
    # Approach #1: No mutation vs. mutation (SNP/INS/DEL)

    # Get path
    mut_path = Path(TCGA_PATHS["MUT"])

    # Load data
    df_mut = pd.read_csv(mut_path, sep="\t")

    # Remove low impact mutations
    df_mut = df_mut[(df_mut.IMPACT != "LOW")]

    # No distinction between mutation types
    df_mut["Variant_Type"] = df_mut.Variant_Type.map({
        "SNP": 1,
        "DEL": 1,
        "INS": 1,
    })

    # Pivot table
    df_mut = df_mut.drop_duplicates(subset=["Sample_short", "Hugo_Symbol"])
    df_mut = df_mut.pivot(index="Sample_short", columns="Hugo_Symbol", values="Variant_Type")
    df_mut = df_mut.fillna(0)
    df_mut = df_mut.astype(int)
    
    # Convert to proper format
    df_mut = df_mut.reset_index()
    df_mut = df_mut.rename(columns={"Sample_short": "sample_id"})

    df_mut["patient_id"] = df_mut.sample_id.apply(lambda x: x[:12])

    # Approach #2: No mutation vs. snip vs. insertion vs. deletion
    #### TODO

    return df_mut


def load_clin(
    cohort: str,
) -> pd.DataFrame:
    """
    Load TCGA clinical data as a pandas dataframe.

    Parameters
    ----------
    cohort: str
        Name of the TCGA cohort.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with columns [
            "patient_id",
            "age", "gender", "race", "grade", "stage",
            "OS_event", "OS_time", "PFS_event", "PFS_time",
        ].
        Id is "patient_id".
    """
    # Get path
    clin_path = Path(TCGA_PATHS["CLINICAL"])

    # Load data
    df_clin = pd.read_excel(clin_path, index_col=0)

    # Select cohort of interest
    df_clin = df_clin[(df_clin.type == cohort)]

    # Get proper names to columns
    to_rename = {
        "bcr_patient_barcode": "patient_id",
        "age_at_initial_pathologic_diagnosis": "age",
        "gender": "gender",
        "race": "race",
        "histological_grade": "grade",
        "ajcc_pathologic_tumor_stage": "stage",
        "OS": "OS_event",
        "OS.time": "OS_time",
        "PFI": "PFS_event",
        "PFI.time": "PFS_time",
    }
    df_clin = df_clin.rename(columns=to_rename)

    # Replace invalid values by NaN
    df_clin = df_clin.replace("[Not Available]", None)

    # Only keep relevant columns
    to_keep = [
        "patient_id",
        "age",
        "gender",
        "race",
        "grade",
        "stage",
        "OS_event",
        "OS_time",
        "PFS_event",
        "PFS_time",
    ]
    df_clin = df_clin[to_keep]

    return df_clin
