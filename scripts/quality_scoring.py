from openai import OpenAI, AsyncOpenAI
import asyncio
import json 
import pandas as pd
from tqdm import tqdm
from collections import Counter
async_client = AsyncOpenAI()

# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================

HYDROLOGY_COMPONENTS = {
    "Water Use & Demand": "Project water abstraction, consumption, recirculation, and discharge — volumes, sources, and water balance",
    "Water Quality": "Physical, chemical, and biological characteristics of water — baseline parameters, effluent quality, receiving water standards, contamination risk",
    "Stormwater": "Rainfall-runoff processes, drainage, flood risk, erosion and sediment transport — not wildfire or vegetation water stress",
    "Groundwater": "Aquifer characteristics, groundwater levels, recharge/discharge zones, groundwater-surface water connectivity, contamination pathways",
    "Surface Water": "Flow regime, catchment hydrology, seasonal variability, flood frequency, channel morphology, aquatic habitat",
    "Social": "Social dimensions of water — community access, rights, downstream security, and human rights implications of project water interactions",
}


ASSESSMENT_CRITERIA = {
    "description_of_water_use": "Information on the project's use and emissions of water — volumes, sources, treatment, discharge",
    "scoping":                  "Whether the EIA defines the spatial and temporal boundaries of water-related assessment — study area, area of influence, and which water bodies or receptors are included or excluded",
    "baseline_study":           "Types, sources and comprehensiveness of baseline data on the aquatic environment — flow, quality, groundwater levels",
    "impact_identification":    "Methods used to identify water-related impacts — pathways, receptors, spatial extent",
    "significance_evaluation":  "Whether water impact significance is evaluated — magnitude, duration, reversibility, criteria used",
    "impact_mitigation":        "Proposed measures to reduce or avoid water-related impacts — design standards, monitoring, contingency",
    "alternatives":             "Whether alternative approaches to water use or management were considered",
}

CRITERIA_SYSTEM_PROMPT = """
You are evaluating pages from a Chilean Environmental Impact Assessment (EIA) for a 
rare earth clay surface mining project. Pages are in Spanish.

Your task is to score how well each page addresses specific EIA assessment criteria
in relation to water and hydrology.

Scoring scale:
0 = Not present — criterion not addressed at all
1 = Present but vague or generic — mentioned but no data, units, or site-specific detail
2 = Present and partially specific — some data OR methodology, but incomplete
3 = Present and fully specific — quantified AND site-specific AND methodology stated

CRITICAL BOUNDARY RULES:
- A score of 1 requires only a mention. If the text says "water impacts will be monitored" with no method → 1.
- A score of 2 requires at least ONE of: a number with units, a named methodology, or a site-specific measurement.
- A score of 3 requires ALL THREE: quantified values + site-specific context + stated methodology.
- When uncertain between 1 and 2, ask: "Is there a number or a named method?" If no → 1.
- Blank forms, templates, and control sheets with empty fields score 0-1 at most. Document titles or footers referencing a plan do not constitute mitigation content on the page. Labels are not data.
- Water management practices cited as justifications ("the project will not be affected because it recirculates water") score 1 for impact_mitigation, not 2. A mitigation measure must explicitly aim to reduce an identified impact.
- Do not let the justification rule suppress scores on other criteria. If a page contains a genuine mention of baseline data or impact significance — even thin — score it accordingly. The justification rule applies only to impact_mitigation.
- Pages containing only climate trend analysis, index definition tables, or statistical significance testing score 0 for impact_mitigation. A future intent to monitor ("debe ser constatado") is not a mitigation measure.
- significance_evaluation requires EIA-specific criteria (magnitude, duration, reversibility applied to a receptor). Statistical significance of climate trends alone does not qualify.
- impact_mitigation scores ≥ 1 only if a measure is proposed in response to an identified impact. If a design feature (e.g. water recirculation, treatment plant sourcing) is cited solely to argue the project will not cause impacts — without referencing what impact it reduces or what the baseline condition is — score impact_mitigation 0 and impact_identification 0. This is a justification, not an assessment.
- impact_mitigation scores ≥ 1 ONLY if the page proposes a future action or operational control to reduce a water-related impact. Past activities (baseline studies, completed drilling, historical monitoring) score 0 for impact_mitigation, regardless of whether they enable future work.
- Do not extract mitigation measures from descriptions of baseline characterization activities, even if those activities could support future monitoring.
- impact_identification scores ≥ 1 ONLY if the page describes a causal pathway: (1) a specific project action or design feature, (2) how it interacts with a water component, and (3) the resulting change or stress to that water component. Baseline findings (e.g., "the aquifer has low productivity") are not impacts; they are context. Score baseline findings as baseline_study, not impact_identification. An impact requires a project nexus.
- Do not extract as impact_identification: aquifer characteristics discovered during baseline study, historical water availability data, or measurements of existing conditions. These describe what is already there, not what the project changes.
- Pages describing only baseline study results, characterization data, or measurements of baseline conditions score impact_identification = 0, even if those results are unfavorable (low yields, poor water quality, etc.).
- A page that contains ONLY a justification claim ("el proyecto usa agua de planta de tratamiento y por tanto no se verá afectado") with no baseline, no impact pathway, and no receptor identified scores 0 on both impact_identification and impact_mitigation.
- Climate change impact discussions that name a hazard type and a hydrological mechanism (e.g. "concentrated extreme rainfall events could cause flooding and mass movement") qualify for impact_identification = 1 even without a named project receptor, provided the link to hydrological conditions is explicit.

Return ONLY valid JSON. No preamble, no markdown fences.
"""

SUBTOPIC_SYSTEM_PROMPT = """
You are analyzing pages from a Chilean Environmental Impact Assessment (EIA) for a 
rare earth clay surface mining project.

You will be given a page text and a set of hydrology topic scores with rationales.
Your task is to identify which specific subtopics are present for each topic that scored >= 1.

Return ONLY valid JSON. No preamble, no markdown fences.
"""

CRITERIA_TO_TAG = set(ASSESSMENT_CRITERIA.keys())

def build_criteria_scoring_prompt(page_text: str) -> str:
    criteria_str = "\n".join([f"- {k}: {v}" for k, v in ASSESSMENT_CRITERIA.items()])
    return f"""
The following page is extracted from a Chilean EIA document written in Spanish.
Score how well it addresses each of the EIA assessment criteria listed below,
in relation to water and hydrology only.

Assessment criteria:
{criteria_str}

For each criterion, assign a score 0-3:
0 = Not present
1 = Present but vague or generic — mention only, no data or named method
2 = Present and partially specific — ONE of: a number, a named method, or a site-specific measurement
3 = Present and fully specific — ALL of: quantified + site-specific + methodology stated

PAGE TEXT (Spanish):
{page_text}

Respond with JSON in this exact structure:
{{
  "scores": {{
    "description_of_water_use": {{"score": <0-3>, "rationale": "<max 20 words, in English>"}},
    "scoping":                  {{"score": <0-3>, "rationale": "<max 20 words, in English>"}},
    "baseline_study":           {{"score": <0-3>, "rationale": "<max 20 words, in English>"}},
    "impact_identification":    {{"score": <0-3>, "rationale": "<max 20 words, in English>"}},
    "significance_evaluation":  {{"score": <0-3>, "rationale": "<max 20 words, in English>"}},
    "impact_mitigation":        {{"score": <0-3>, "rationale": "<max 20 words, in English>"}},
    "alternatives":             {{"score": <0-3>, "rationale": "<max 20 words, in English>"}}
  }}
}}
"""

async def tag_components_for_criterion(
    row: pd.Series,
    criterion: str,
    score: float,
    rationale: str,
    text_col: str = "spanish_text"
) -> dict:
        if criterion == "baseline_study":
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on baseline_study (characterisation of baseline water conditions).
Scoring rationale: {rationale}

Identify which hydrological components are covered as baseline data in the page text.
Only include components that are genuinely present. Omit absent ones.

IMPORTANT DISTINCTIONS:
- Precipitation trend data, rainfall statistics, and flow regime data belong to "Surface Water" unless the page explicitly discusses drainage infrastructure, runoff management, or erosion control — in which case also include "Stormwater".
- "Stormwater" requires content about drainage systems, runoff management, or flood risk to infrastructure — not just precipitation data.
- "Social" requires explicit reference to community water access, rights, or downstream users — not general climate vulnerability.

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<max 20 words in English>"}}
}}
"""

        elif criterion == "impact_identification":
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on impact_identification (identification of water-related impacts).
Scoring rationale: {rationale}

Identify which hydrological components have water-related impacts identified on this page.
Only include components whose impacts are genuinely identified. Omit absent ones.
In each rationale, describe the specific impact to that component (e.g. "reduced downstream flow due to abstraction", "groundwater drawdown from pit dewatering").

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<specific impact identified, max 20 words in English>"}}
}}
"""

        elif criterion == "impact_mitigation":
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on impact_mitigation (proposed measures to reduce water-related impacts).
Scoring rationale: {rationale}

Identify which hydrological components have mitigation measures explicitly proposed on this page.
A mitigation measure must genuinely aim to reduce an identified impact to that component.

Include: design features that actively reduce water impacts (e.g. recirculation reducing abstraction, sediment ponds reducing turbidity).
Exclude: assertions that components "will not be affected" without proposing measures (e.g. "the project uses recirculated water therefore water quality will not be affected") — these justify not assessing impact rather than mitigating it.

Only include components with genuine mitigation measures present. Omit absent ones.
In each rationale, describe the specific measure targeting that component (e.g. "water recirculation system to reduce abstraction", "sediment retention pond").

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<specific measure proposed, max 20 words in English>"}}
}}
"""
        elif criterion == "description_of_water_use":
            # Same hydrology components as baseline_study
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on description_of_water_use (describing project water use).
Scoring rationale: {rationale}

Identify which hydrological components are addressed in the page's description of project water use.
Only include components that are genuinely present. Omit absent ones.

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<max 20 words in English>"}}
}}
"""

        elif criterion == "scoping":
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on scoping (defining spatial and temporal boundaries of water assessment).
Scoring rationale: {rationale}

Identify which hydrological components have spatial or temporal boundaries, study areas, water bodies, or receptors defined on this page.
Only include components that are genuinely scoped. Omit absent ones.
In each rationale, state the specific boundary/area defined (e.g. "mine site boundary", "downstream 50 km", "aquifer extent", "5-year assessment period").

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<specific boundary or receptor, max 20 words in English>"}}
}}
"""

        elif criterion == "significance_evaluation":
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on significance_evaluation (evaluating water impact significance).
Scoring rationale: {rationale}

Identify which hydrological components have significance criteria, methods, or thresholds applied on this page.
Only include components genuinely evaluated. Omit absent ones.
In each rationale, state the specific criterion/method/threshold applied (e.g. "magnitude assessment", "duration", "reversibility criteria", specific thresholds).

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<specific criterion or method, max 20 words in English>"}}
}}
"""

        elif criterion == "alternatives":
            components_str = "\n".join([
                f"- {component}: {definition}"
                for component, definition in HYDROLOGY_COMPONENTS.items()
            ])
            prompt = f"""
This page scored {score} on alternatives (considering alternative approaches to water use or management).
Scoring rationale: {rationale}

Identify which hydrological components are addressed by alternative approaches, scenarios, or options considered on this page.
Only include components genuinely covered by an alternative. Omit absent ones.
In each rationale, state the specific alternative (e.g. "alternative water sources", "alternative discharge methods", "alternative pit locations").

Available hydrological components:
{components_str}

PAGE TEXT (Spanish):
{row[text_col]}

Respond with JSON where keys are component names from the list above:
{{
  "<component_name>": {{"rationale": "<specific alternative, max 20 words in English>"}}
}}
"""

        try:
            r = await async_client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SUBTOPIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            tqdm.write(f"Component tagging error on page {row.get('page')} / {criterion}: {e}")
            return {}


async def score_page_criteria_with_components(row: pd.Series, semaphore: asyncio.Semaphore, text_col: str = "spanish_text") -> list:
    async with semaphore:
        try:
            responses = await asyncio.gather(*[
                async_client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": CRITERIA_SYSTEM_PROMPT},
                        {"role": "user",   "content": build_criteria_scoring_prompt(row[text_col])},
                    ],
                )
                for _ in range(3)
            ])
            all_scores = [json.loads(r.choices[0].message.content)["scores"] for r in responses]

            majority_scores = {}
            for criterion in all_scores[0].keys():
                reviewer_scores = [all_scores[i][criterion]["score"] for i in range(3)]
                majority_scores[criterion] = {
                    "score":    round(sum(reviewer_scores) / 3, 2),
                    "rationale": all_scores[0][criterion]["rationale"],
                }

            # Run component tagging independently per criterion
            component_results = {}
            for criterion in CRITERIA_TO_TAG:
                reviewer_scores_for_criterion = [all_scores[i][criterion]["score"] for i in range(3)]
                if any(s >= 1 for s in reviewer_scores_for_criterion):
                    component_results[criterion] = await tag_components_for_criterion(
                        row=row,
                        criterion=criterion,
                        score=majority_scores[criterion]["score"],
                        rationale=majority_scores[criterion]["rationale"],
                        text_col=text_col,
                    )

            records = []
            for criterion in all_scores[0].keys():
                reviewer_scores = [all_scores[i][criterion]["score"] for i in range(3)]
                mean_score = round(sum(reviewer_scores) / 3, 2)
                std_score  = round((sum((s - mean_score) ** 2 for s in reviewer_scores) / 3) ** 0.5, 2)

                comp = component_results.get(criterion, {})
                records.append({
                    "file_es":             row["file_es"],
                    "file_en":             row["file_en"],
                    "page":                row["page"],
                    "content_es":          row["spanish_text"],
                    "content_en":          row["english_text"],
                    "criterion":           criterion,
                    "score_r1":            reviewer_scores[0],
                    "score_r2":            reviewer_scores[1],
                    "score_r3":            reviewer_scores[2],
                    "score_mean":          mean_score,
                    "score_std":           std_score,
                    "high_uncertainty":    std_score >= 0.75,
                    "rationale_r1":        all_scores[0][criterion]["rationale"],
                    "rationale_r2":        all_scores[1][criterion]["rationale"],
                    "rationale_r3":        all_scores[2][criterion]["rationale"],
                    "components":          list(comp.keys()),
                    "component_rationale": json.dumps({k: v.get("rationale", "") for k, v in comp.items()}),
                })
            return records

        except Exception as e:
            tqdm.write(f"Error on {row.get('page')}: {e}")
            return []

async def score_all_pages_criteria_with_components(df: pd.DataFrame, concurrency: int = 10) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [score_page_criteria_with_components(row, semaphore) for _, row in df.iterrows()]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.extend(await coro)
    return pd.DataFrame(results)
