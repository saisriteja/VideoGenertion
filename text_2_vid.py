import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Load the model
pipe = DiffusionPipeline.from_pretrained("zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# Memory optimizations
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

# Scenarios and mapping
data = {
    "ResidentialAreas": {
        "scenarios": [
            "CCTV footage captures a burglary when a thief breaks into a house through an unlocked window.",
            "A vandal damages public property in a residential area, with the act recorded on CCTV.",
            "CCTV cameras record an intruder attempting unauthorized access to a gated residential community.",
            "A group of teenagers engaging in illegal activities at a park is observed through surveillance footage.",
            "A suspicious vehicle parked near a residential area for an extended period is flagged by CCTV.",
            "An emergency medical response is triggered after CCTV monitors a health incident in a residential area.",
            "CCTV helps locate a child trapped in a house during a fire, enabling a swift rescue.",
            "A neighbor reports an unfamiliar person acting suspiciously, with their movements captured on CCTV.",
            "A domestic dispute escalating to police intervention is partially recorded on neighborhood surveillance.",
            "A power outage causing confusion is monitored by CCTV in a densely populated residential area."
        ],
        "image": "ResidentialAreas/001.png"
    },
    "CommercialProperties": {
        "scenarios": [
            "An employee caught stealing products from a retail store is identified through CCTV footage.",
            "CCTV cameras record a customer involved in a shoplifting incident at a commercial property.",
            "Unauthorized access to a restricted area within a mall is flagged by surveillance cameras.",
            "CCTV aids in investigating a corporate espionage case in an office building.",
            "A security guard responds to an attempted theft observed on CCTV at a commercial property.",
            "A robbery in a store inside a commercial complex is recorded on surveillance footage.",
            "Suspicious individuals near an ATM in a commercial area are monitored by CCTV.",
            "An emergency response to a fire in a shopping mall is informed by CCTV footage.",
            "A violent altercation between employees in a commercial office is captured on surveillance cameras.",
            "Fraud at a retail store’s point of sale is discovered through CCTV monitoring."
        ],
        "image": "commercial_complex/001.png"
    },
    "PublicSpaces": {
        "scenarios": [
            "Disorderly conduct at a public event is captured on CCTV cameras.",
            "A suspicious package discovered in a crowded public space is flagged by surveillance footage.",
            "CCTV monitors a protest escalating into violent riots in a public square.",
            "A fight breaking out at a bus stop is observed on public surveillance cameras.",
            "Overcrowding concerns in a public park are assessed using CCTV footage.",
            "Graffiti vandalism on a public building is recorded by nearby surveillance cameras.",
            "A car accident causing a traffic jam in a busy public square is captured on CCTV.",
            "A suspicious person loitering around public benches is flagged by surveillance footage.",
            "A stampede during a panic situation is observed through public CCTV systems.",
            "An emergency medical situation in a crowded public area is monitored by CCTV cameras."
        ],
        "image": "environment_Surveillance/001.png"
    },
    "HighwaysandIntersections": {
        "scenarios": [
            "CCTV footage records a car accident causing a significant traffic jam on a busy highway.",
            "Traffic cameras capture drivers ignoring signals, leading to a potential collision at an intersection.",
            "A traffic violation during rush hour is flagged by intersection surveillance.",
            "CCTV cameras observe pedestrians jaywalking across a major highway, risking accidents.",
            "A road rage incident at a highway junction is recorded on surveillance footage.",
            "An emergency vehicle stuck in traffic on a highway is monitored through CCTV systems.",
            "A vehicle running a red light, causing a near-miss with pedestrians, is captured on traffic cameras.",
            "Illegal blockage of an intersection by a construction vehicle is flagged by CCTV.",
            "Traffic signs being ignored at a highway on-ramp, leading to an accident, is recorded on surveillance footage.",
            "A hit-and-run incident at a busy intersection is captured by CCTV cameras during peak hours."
        ],
        "image": "HighwaysandIntersections/001.png"
    },
    "PublicTransport": {
        "scenarios": [
            "Harassment of an individual on a crowded bus is captured on public transport CCTV.",
            "A suspicious package found on a train is flagged through surveillance footage.",
            "Overcrowding in a subway station causing panic is monitored by CCTV cameras.",
            "CCTV cameras record a fight breaking out between two passengers on public transport.",
            "Theft of personal items during a crowded commute is identified through surveillance footage.",
            "An emergency medical situation on a bus is flagged by onboard CCTV cameras.",
            "Suspicious activity at a train station is recorded on CCTV.",
            "CCTV cameras capture an elderly passenger collapsing in a crowded subway.",
            "A person causing a disruption on public transport is monitored through surveillance footage.",
            "A traffic accident delaying public transport services is observed on CCTV systems."
        ],
        "image": "HighwaysandIntersections/001.png"
    },
    "FactoriesandWarehouses": {
        "scenarios": [
            "CCTV footage captures a worker accidentally touching molten metal, risking severe burns.",
            "An operator's hand caught between moving machinery is recorded by factory surveillance.",
            "A worker slipping on an oil spill in the workshop is flagged by CCTV cameras.",
            "CCTV monitors an employee suffering a deep cut while handling sharp metal sheets.",
            "Flying debris during welding injures unprotected eyes, captured on factory CCTV footage.",
            "CCTV observes a worker inhaling toxic fumes from welding without proper ventilation.",
            "Noise exposure near loud machinery without ear protection is recorded by surveillance.",
            "A technician receiving an electric shock during repairs is captured on CCTV.",
            "CCTV cameras monitor a chemical spill causing burns on a worker’s arm and face.",
            "An employee lifting heavy metal beams experiences back strain, recorded on surveillance footage."
        ],
        "image": "FactoriesandWarehouses/001.png"
    },
    "ConstructionSites": {
        "scenarios": [
            "A worker falling from scaffolding at a construction site is captured on CCTV footage.",
            "Unauthorized workers accessing a restricted construction zone are flagged by surveillance.",
            "CCTV cameras monitor a worker injured by machinery malfunction on the site.",
            "Mishandling of heavy equipment causing damage is recorded by site surveillance.",
            "A worker suffering a head injury from falling debris is observed on CCTV footage.",
            "CCTV monitors a trench collapse trapping a worker at a construction site.",
            "Neglect of protective equipment by workers is flagged by construction site surveillance.",
            "CCTV records a crane malfunction causing debris to fall at a construction site.",
            "An electrical shock injury during machinery operation is captured on site surveillance footage.",
            "A collision between construction trucks causing injuries is recorded on CCTV cameras."
        ],
        "image": "ConstructionSites/001.png"
    },
    "CrimeInvestigation": {
        "scenarios": [
            "CCTV footage aids police in investigating a break-in at a residential property.",
            "A forensic team examines robbery traces recorded by surveillance footage.",
            "Surveillance footage is analyzed to investigate a theft in a retail store.",
            "CCTV data is reviewed by detectives to track the movements of a suspect.",
            "Evidence collected at a public park after an assault incident includes CCTV footage.",
            "A crime scene investigation team uses surveillance footage from a construction site.",
            "A security camera captures an individual committing fraud at an ATM.",
            "CCTV assists detectives in tracking details during a vehicle theft investigation.",
            "A security guard reviews video footage to identify potential suspects in a break-in.",
            "Forensic experts analyze digital traces supported by surveillance footage in a fraud case."
        ],
        "image": "commercial_complex/001.png"
    },
    "CrowdManagement": {
        "scenarios": [
            "CCTV monitors a panic situation requiring crowd control at a public event.",
            "A stampede during a concert is captured on surveillance cameras.",
            "CCTV systems aid security teams in managing large crowds at a sports event.",
            "Crowds overflowing beyond designated areas are flagged by surveillance footage.",
            "CCTV monitors crowds gathering around an emergency scene, obstructing rescue operations.",
            "Security measures during a protest are informed by real-time CCTV footage.",
            "Overcrowding at public entrances causing chaos is captured on surveillance cameras.",
            "CCTV systems monitor dangerous crowds at a political rally.",
            "Barriers controlling crowds during a street festival are assessed through CCTV footage.",
            "Crowd management during a religious gathering is supported by CCTV monitoring."
        ],
        "image": "commercial_complex/001.png"
    }
}

# Define the output path
model_name = "zeroscope_v2_576w"
output_folder = f"output_cctv/{model_name}"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each category and its scenarios
for category, details in data.items():
    for i, prompt in enumerate(details["scenarios"]):
        print(f"Processing: {category} - {prompt}")
        
        # Generate video frames
        video_frames = pipe(prompt, num_frames=24).frames[0]
        
        # Create a category-specific folder
        category_folder = os.path.join(output_folder, f"{category}")
        os.makedirs(category_folder, exist_ok=True)
        
        # Save the video to a file
        sanitized_prompt = prompt.replace(" ", "_").replace(",", "").replace(".", "").lower()[:50]
        video_path = os.path.join(category_folder, f"{i+1:02d}_{sanitized_prompt}.mp4")
        export_to_video(video_frames, video_path)
        print(f"Saved video to {video_path}")
