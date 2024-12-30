import os
import random
from diffusers import I2VGenXLPipeline
from PIL import Image
import torch
import numpy as np
import shutil
from tqdm import tqdm
from diffusers.utils import export_to_video

# Embed the scenarios and image mapping as a nested dictionary
data = {
    "ResidentialAreas": {
        "scenarios": [
            "A burglary occurs when a thief breaks into a house through an unlocked window.",
            "A vandal damages a public property in a residential area, causing significant harm.",
            "An intruder attempts unauthorized access to a gated residential community.",
            "A group of teenagers engages in illegal activities at a park in a residential neighborhood.",
            "A suspicious vehicle is parked near a residential area for an extended period.",
            "An emergency medical response is required due to a health incident in a residential area.",
            "A child gets trapped in a house during a fire, requiring immediate rescue.",
            "A neighbor reports an unfamiliar person acting suspiciously in the neighborhood.",
            "A domestic dispute escalates, leading to police intervention.",
            "A power outage causes confusion and chaos in a densely populated residential area."
        ],
        "image": "ResidentialAreas/001.png"
    },
    "CommercialProperties": {
        "scenarios": [
            "An employee is caught stealing products from a retail store.",
            "A customer is involved in a shoplifting incident at a commercial property.",
            "An unauthorized person gains access to a restricted area within a mall.",
            "A corporate espionage case is being investigated in an office building.",
            "A security guard responds to an attempted theft at a commercial property.",
            "A robbery occurs in a store located inside a commercial complex.",
            "Suspicious individuals are spotted near an ATM in a commercial area.",
            "An emergency response team is needed due to a fire in a shopping mall.",
            "A dispute between employees turns into a violent altercation in a commercial office.",
            "A customer reports fraud occurring at a retail store’s point of sale."
        ],
        "image": "commercial_complex/001.png"
    },
    "PublicSpaces": {
        "scenarios": [
            "A group of individuals engages in disorderly conduct at a public event.",
            "A suspicious package is discovered in a crowded public space.",
            "A protest escalates into violent riots in a public square.",
            "A fight breaks out between two individuals at a bus stop.",
            "A crowd gathers in a public park, causing overcrowding concerns.",
            "A vandal spray-paints graffiti on a public building.",
            "A car accident causes a traffic jam in a busy public square.",
            "A person is seen loitering suspiciously around public benches.",
            "A public gathering leads to a stampede during a panic situation.",
            "An emergency medical situation arises in a densely populated public area."
        ],
        "image": "environment_Surveillance/001.png"
    },
    "HighwaysandIntersections": {
        "scenarios": [
            "A car accident causes a significant traffic jam on a busy highway.",
            "Drivers ignore traffic signals, leading to a potential collision at an intersection.",
            "A traffic violation is recorded at a busy intersection during rush hour.",
            "Pedestrians are seen jaywalking across a major highway, risking accidents.",
            "A road rage incident occurs between two drivers at a highway junction.",
            "An emergency vehicle is stuck in traffic on a highway due to congestion.",
            "A vehicle runs a red light, causing a near-miss with pedestrians.",
            "A construction vehicle illegally blocks an intersection, disrupting traffic flow.",
            "Traffic signs are ignored, leading to an accident at a highway on-ramp.",
            "A hit-and-run incident occurs at a busy intersection during peak traffic hours."
        ],
        "image": "HighwaysandIntersections/001.png"
    },
    "PublicTransport": {
        "scenarios": [
            "An individual is harassed by a group of people on a crowded bus.",
            "A suspicious package is found on a train, causing an emergency evacuation.",
            "Overcrowding in a subway station causes panic among commuters.",
            "A fight breaks out between two passengers on a public transport vehicle.",
            "A passenger reports theft of personal items during a crowded commute.",
            "An emergency medical situation arises on a bus during rush hour.",
            "A train conductor notices suspicious activity in a station.",
            "An elderly passenger collapses in a crowded subway.",
            "A person attempts to cause a disruption on a public transport system.",
            "A traffic accident delays public transport services during peak hours."
        ],
        "image": "HighwaysandIntersections/001.png"
    },
    "FactoriesandWarehouses": {
        "scenarios": [
            "A worker accidentally touches molten metal, risking severe burns and scalds.",
            "An operator's hand gets caught between moving machinery parts, causing a crushing injury.",
            "A worker slips on an oil spill in the workshop, leading to a potential fall hazard.",
            "An employee handling sharp metal sheets suffers a deep cut on their hand.",
            "A welder's unprotected eyes are injured by flying metal debris and sparks.",
            "A worker inhales toxic fumes from welding without proper ventilation, endangering respiratory health.",
            "An employee works near loud machinery without ear protection, risking permanent hearing loss.",
            "A technician suffers an electric shock due to contact with a live wire during repairs.",
            "A worker accidentally spills a chemical, causing burns on their arm and face.",
            "An employee lifting heavy metal beams experiences severe back strain and discomfort."
        ],
        "image": "FactoriesandWarehouses/001.png"
    },
    "ConstructionSites": {
        "scenarios": [
            "An accident occurs when a worker falls from scaffolding at a construction site.",
            "Unauthorized workers gain access to a restricted construction zone.",
            "A worker is injured when machinery malfunctions on the site.",
            "Heavy equipment is mishandled, causing damage and injury to workers.",
            "A construction worker suffers a head injury from falling debris.",
            "A worker in a trench is trapped due to a collapse at a construction site.",
            "A safety hazard arises when workers neglect proper protective equipment.",
            "An incident happens when a crane malfunctions, causing debris to fall.",
            "A worker suffers an electrical shock while operating machinery at a site.",
            "A vehicle collision occurs on-site between construction trucks, causing injuries."
        ],
        "image": "ConstructionSites/001.png"
    },
    "CrimeInvestigation": {
        "scenarios": [
            "Police investigate a break-in at a residential property, gathering evidence.",
            "A forensic team examines a crime scene for traces of a robbery.",
            "Surveillance footage is analyzed to investigate a theft in a retail store.",
            "A detective reviews camera data to track the movements of a suspect.",
            "Evidence is collected at a public park after an assault incident.",
            "A crime scene investigation team works at a construction site to uncover details of a theft.",
            "A security camera captures an individual committing fraud at an ATM.",
            "Detectives look for clues during a vehicle theft investigation.",
            "A security guard reviews video footage to identify potential suspects in a break-in.",
            "Forensic experts analyze digital footprints in a fraud case investigation."
        ],
        "image": "commercial_complex/001.png"
    },
    "CrowdManagement": {
        "scenarios": [
            "A panic situation arises at a public event, requiring crowd control.",
            "A stampede occurs during a concert, endangering attendees.",
            "Security teams manage large crowds during a sports event.",
            "A crisis develops when the crowd overflows beyond the designated area.",
            "A crowd gathers around an emergency scene, obstructing rescue operations.",
            "Security measures are enforced during a protest to control the crowd.",
            "A major event leads to overcrowding at public entrances, causing chaos.",
            "Authorities handle a dangerous crowd during a political rally.",
            "Police use barriers to control a large crowd during a street festival.",
            "Crowd management becomes critical during a religious gathering in a public space."
        ],
        "image": "commercial_complex/001.png"
    }
}


# Define paths
negative_prompt = "Distorted"
output_dir = "output_vids"
model_name = "i2vgen-xl"
main_path = '/home/user/Pictures/'

# Ensure output directory exists
os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)

# Cache file to track progress
cache_file = 'progress_cache.txt'

# Helper function to load image
def load_image(image_path):
    return Image.open(image_path)

# Load cache from the last progress checkpoint, if available
def load_cache():
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return f.read().strip()
    return None

# Save the current progress to cache
def save_cache(progress):
    with open(cache_file, 'w') as f:
        f.write(progress)

# Get last processed category and caption from the cache
last_processed = load_cache()

# Iterate through the database to generate videos for each scenario
for category, category_data in tqdm(data.items(), desc="Categories"):

    try:

        img_path = category_data["image"]
        img_path = os.path.join(main_path, img_path)
        
        # Load image
        image = load_image(img_path)
        
        # Convert to RGB
        image = image.convert("RGB")
        
        # Iterate through captions and process them
        for i, caption in tqdm(enumerate(category_data["scenarios"]), desc="Captions", leave=False):
            # Skip previously processed captions within the current category
            if last_processed and category == last_processed.split('_')[0] and i < int(last_processed.split('_')[1]):
                continue

            # Load the model pipeline
            pipeline = I2VGenXLPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            pipeline.enable_model_cpu_offload()

            prompt = caption
            generator = torch.manual_seed(8888)

            # Generate frames
            frames = pipeline(
                prompt=prompt,
                image=image,
                num_inference_steps=50,
                negative_prompt=negative_prompt,
                guidance_scale=9.0,
                generator=generator
            ).frames[0]
            
            # Construct output path
            img_name = os.path.basename(img_path).split('.')[0]
            caption_name = caption.replace(' ', '_').replace(',', '').replace('.', '')
            category_name = category.replace(' ', '_')
            output_path = os.path.join(output_dir, model_name, f"{category_name}_{img_name}_{caption_name}.mp4")
            
            # Export frames to video
            export_to_video(frames, output_path)
            
            # Update cache after processing each caption
            # save_cache(f"{category}_{i+1}")
        

    except Exception as e:
        print(f"An error occurred: {e}")
        # Update cache with the last processed caption in case of an error
        # save_cache(f"{category}_{i}")

    # After processing all captions for the category, update cache with the category name
    # save_cache(category)