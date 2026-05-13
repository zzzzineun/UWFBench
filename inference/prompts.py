SYSTEM_PROMPT = """You are an expert ophthalmologist generating reports for Ultra-Widefield (UWF) fundus images.

Rules:
- Describe only what is visually verifiable in the pixels.
- Do NOT mention anatomical structures (e.g., Optic Disc, Macula, Fovea, Vessels, or Periphery) unless they are clearly visible.
- Do NOT use placeholder phrases like "within normal limits," "unremarkable," or "appears normal" for structures not being described. If you don't see it, don't write it.
- Do not explain the pathophysiology or define diseases. Provide only findings and a resulting diagnosis/grade based strictly on visible findings.
- Highly professional, objective, and concise. Use standard ophthalmic nomenclature.
"""




USER_PROMPTS = {


    "report_v1": {
        "output_prefix": "./results/inference1",
        "text": """
Generate a concise ophthalmology-style report (1 paragraph, 1–3 sentences) for the provided Ultra-Widefield (UWF) fundus image.

Requirements:
- Describe ONLY findings that are directly visible in the image.
- Do NOT infer, speculate, or include unseen/assumed findings.
- Do NOT include differential diagnoses.
- Do not mention unseen normal findings.
- Maintain internally consistent clinical impression. All described findings must support coherent retinal condition or pattern.
- When present, include:
    (1) main diagnosis or disease pattern with severity (if applicable)
    (e.g., No DR (Diabetic Retinopathy), Mild/Moderate/Severe NPDR (Non-proliferative Diabetic Retinopathy), PDR (Proliferative Diabetic Retinopathy), CRVO (Central Retinal Vein Occlusion)/hemi-CRVO/BRVO (Branch Retinal Vein Occlusion)/RVO (Retinal Vein Occlusion), AMD (Age-related Macular Degeneration), RD (Retinal Detachment), Uveitis, myopic/tigroid fundus, PRP (Panretinal Photocoagulation) scars, optic disc WNL/PPA (Peripapillary Atrophy)/Cup-to-Disc changes)
    (2) key lesions
    (3) retinal location (if identifiable)
- Keep wording consistent with ophthalmology fundus reporting style and use short clinical phrases.

Example Reports (style-only):
- Ultra-widefield fundus photography reveals moderate non-proliferative diabetic retinopathy (NPDR) with hard exudates and hemorrhages present within the macula. A superotemporal retinal nerve fiber layer defect (RNFLD) is noted, and an inferotemporal peripheral choroidal nevus is present.
- Ultra-widefield fundus photography demonstrates findings consistent with severe non-proliferative diabetic retinopathy (NPDR) or proliferative diabetic retinopathy (PDR). A dense ring of hard exudates is present at the temporal macula. Multiple dot and blot hemorrhages and tortuous retinal vessels are noted.
- Ultra-widefield fundus photography demonstrates a retinal detachment involving the macula. The fundus exhibits a tigroid appearance. The optic disc is tilted with associated peripapillary atrophy.
- Ultra-widefield fundus photography demonstrates no evidence of diabetic retinopathy. The foveal reflex is present, and a media opacity is noted at the temporal side.
- Ultra-widefield fundus photography demonstrates a branch retinal vein occlusion (BRVO) with associated retinal hemorrhages, cotton wool spots, and hard exudates involving the superior macula and superotemporal retina. Peripheral laser scars are noted in the superotemporal periphery.
""",
    },



    "report_v2": {
        "output_prefix": "./results/inference2",
        "text": """
Generate a concise ophthalmology-style report (1 paragraph, 1–3 sentences) for the provided Ultra-Widefield (UWF) fundus image.

Requirements:
- Describe ONLY findings that are directly visible in the image.
- Do NOT infer, speculate, or include unseen/assumed findings.
- Do NOT include differential diagnoses.
- Do not mention unseen normal findings.
- Maintain internally consistent clinical impression. All described findings must support coherent retinal condition or pattern.
- When present, include:
    (1) main diagnosis or disease pattern with severity (if applicable)
    (e.g., No DR (Diabetic Retinopathy), Mild/Moderate/Severe NPDR (Non-proliferative Diabetic Retinopathy), PDR (Proliferative Diabetic Retinopathy), CRVO (Central Retinal Vein Occlusion)/hemi-CRVO/BRVO (Branch Retinal Vein Occlusion)/RVO (Retinal Vein Occlusion), AMD (Age-related Macular Degeneration), RD (Retinal Detachment), Uveitis, myopic/tigroid fundus, PRP (Panretinal Photocoagulation) scars, optic disc WNL/PPA (Peripapillary Atrophy)/Cup-to-Disc changes)
    (2) key lesions
    (3) retinal location (if identifiable)
- Keep wording consistent with ophthalmology fundus reporting style and use short clinical phrases.

Example Reports (style-only):
- Ultra-widefield fundus photography reveals moderate non-proliferative diabetic retinopathy (NPDR) with hard exudates and hemorrhages present within the macula. A superotemporal retinal nerve fiber layer defect (RNFLD) is noted, and an inferotemporal peripheral choroidal nevus is present.
- Ultra-widefield fundus photography demonstrates findings consistent with severe non-proliferative diabetic retinopathy (NPDR) or proliferative diabetic retinopathy (PDR). A dense ring of hard exudates is present at the temporal macula. Multiple dot and blot hemorrhages and tortuous retinal vessels are noted.
- Ultra-widefield fundus photography demonstrates a branch retinal vein occlusion (BRVO) with associated retinal hemorrhages, cotton wool spots, and hard exudates involving the superior macula and superotemporal retina. Peripheral laser scars are noted in the superotemporal periphery.
""",
    },




    "report_v3": {
        "output_prefix": "./results/inference3",
        "text": """
Generate a concise ophthalmology-style report (1 paragraph, 1–3 sentences) for the provided Ultra-Widefield (UWF) fundus image.

Requirements:
- Describe ONLY findings that are directly visible in the image.
- Do NOT infer, speculate, or include unseen/assumed findings.
- Do NOT include differential diagnoses.
- Do not mention unseen normal findings.
- Maintain internally consistent clinical impression. All described findings must support coherent retinal condition or pattern.
- When present, include:
    (1) main diagnosis or disease pattern with severity (if applicable)
    (e.g., No DR (Diabetic Retinopathy), Mild/Moderate/Severe NPDR (Non-proliferative Diabetic Retinopathy), PDR (Proliferative Diabetic Retinopathy), CRVO (Central Retinal Vein Occlusion)/hemi-CRVO/BRVO (Branch Retinal Vein Occlusion)/RVO (Retinal Vein Occlusion), AMD (Age-related Macular Degeneration), RD (Retinal Detachment), Uveitis, myopic/tigroid fundus, PRP (Panretinal Photocoagulation) scars, optic disc WNL/PPA (Peripapillary Atrophy)/Cup-to-Disc changes)
    (2) key lesions
    (3) retinal location (if identifiable)
- Keep wording consistent with ophthalmology fundus reporting style and use short clinical phrases.
""",
    }
}
