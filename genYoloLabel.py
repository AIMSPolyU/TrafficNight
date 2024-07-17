import os, json
from pathlib import Path

Dotalabel_mapping = {
    "Car": "Car",
    "Buses": "Buses",
    "Trucks": "Trucks",
    "Semi-Trailers": "Semi-Trailers",
    "Empty Semi-Trailers" : "Empty Semi-Trailers",
}

yololabel_mapping = {
    "Car": "0",
    "Buses": "1",
    "Trucks": "2",
    "Tractor": "3",
    "Semi-Trailers": "4",
    "Empty Semi-Trailers" : "5"
}

def load_labelme_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def save_yolo_obb_format(output_file, label, box, difficulty=10):
    box_str = ' '.join(map(lambda x: str(x), box.flatten()))
    
    with open(output_file, 'a') as file:
        file.write(f"{label} {box_str}\n")

def convert2yolo():
    json_folder = 'TrafficNight/download/'
    output_folder = 'TrafficNight/download/'
    jsonLabel_folder = 'TrafficNight/download/'

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(jsonLabel_folder).mkdir(parents=True, exist_ok=True)
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_path = Path(json_folder, json_file)
        data = load_labelme_json(json_path)
        height, width = data['imageHeight'], data['imageWidth']

        output_file = Path(output_folder, Path(json_file).stem + '.txt')
        output_json = Path(jsonLabel_folder, Path(json_file).stem + '.json')

        for shape in data['shapes']:
            original_label = shape['label']
            corrected_label = yololabel_mapping.get(original_label, original_label)

            save_yolo_obb_format(output_file, corrected_label, shape['points'])
