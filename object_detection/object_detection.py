import torch
import torchvision
import cv2
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

# Load pretarined Faster R-CNN with ResNet50 as backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
model.eval()
print(model)

# Fuction for object detection per frame
def detect_objects(frame, model, threshold=0.5):
    
    # Convert image to tensor
    img_tensor = F.to_tensor(frame).unsqueeze(0)
    
    # Gets model outputs
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Filter predictions higher than a threshold
    pred_boxes = outputs[0]['boxes'][outputs[0]['scores'] > threshold]
    pred_labels = outputs[0]['labels'][outputs[0]['scores'] > threshold]
    pred_scores = outputs[0]['scores'][outputs[0]['scores'] > threshold]
    
    return pred_boxes, pred_labels, pred_scores

# Draw bounding boxes around objects
def draw_boxes(frame, boxes, labels, scores, label_map):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = label_map[labels[i].item()]
        score = scores[i].item()
        text = f"{label}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


LABEL_MAP = {i: label for i, label in enumerate(
    [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
        'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
)}

def run_video():
    # Open video
    PATH = 'video/cow_eating_grass.mp4'
    cap = cv2.VideoCapture(PATH)

    # Gets video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    out = cv2.VideoWriter('./output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Effettua object detection sul frame corrente
        boxes, labels, scores = detect_objects(frame, model)

        # Disegna le bounding box e le label
        frame = draw_boxes(frame, boxes, labels, scores, LABEL_MAP)

        # Salva il frame nel video di output
        out.write(frame)

        # Mostra il frame in tempo reale (opzionale)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def run_image():
    # Path to the image
    IMAGE_PATH = 'img/city.jpeg'

    # Read the image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError(f"Unable to load image at {IMAGE_PATH}")

    
    # Perform object detection
    boxes, labels, scores = detect_objects(image, model)
    
    #print(f"Boxes: {boxes}")
    #print(f"Labels: {labels}")
    #print(f"Scores: {scores}")

    # Draw bounding boxes and labels on the image
    image_bgr = draw_boxes(image, boxes, labels, scores, LABEL_MAP)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    img_pil = Image.fromarray(image_rgb)
    
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()

run_image()
