import json
import numpy as np
import torch 
import torch.nn as nn 
import librosa 
from PIL import Image 
from torchvision import models, transforms 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)
    
def load_models():
    audio_classes = np.load("models/audio_classes.npy", allow_pickle=True)
    audio_model = AudioClassifier(26, len(audio_classes))
    audio_model.load_state_dict(torch.load("models/audio_model.pth", map_location=DEVICE))
    audio_model.to(DEVICE).eval()
    
    image_classes = np.load("models/imagine_classes.npy", allow_pickle=True)
    image_model = models.resnet18(weights=None)
    image_model.fc = nn.Linear(image_model.fc.in_features, len(image_classes))
    image_model.load_state_dict(torch.load("models/image_model.pth", map_location=DEVICE))
    image_model.to(DEVICE).eval()
    
    return audio_model, audio_classes, image_model, image_classes

def extract_audio_features(path, n_mfcc = 13):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis =1)
    return np.concatenate([mfcc_mean, mfcc_std])

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_audio_probs(model, classes, audio_path):
    features = extract_audio_features(audio_path)
    tensor = torch.tensor(features).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

def predict_image_probs(model, classes, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs 

def late_fusion(probs_audio, classes_audio, probs_image, classes_image, alpha=0.5):
    clase_comune = sorted(set(classes_audio) & set(classes_image))
    
    combined_probs = []
    for cls in clase_comune:
        idx_audio = list(classes_audio).index(cls)
        idx_image = list(classes_image).index(cls)
        p = alpha * probs_audio[idx_audio] + (1- alpha) * probs_image[idx_image]
        combined_probs.append(p)
        
    combined_probs = np.array(combined_probs)
    pred_idx = np.argmax(combined_probs)
    return clase_comune[pred_idx]

def evaluate():
    with open("test_perechi.json", "r", encoding="utf-8") as f:
        perechi = json.load(f)
        
    audio_model, audio_classes, image_model, image_classes = load_models()
    
    y_true = []
    y_pred_audio = []
    y_pred_image = []
    y_pred_fusion = []
    
    for pereche in perechi:
        adevarat = pereche["instrument"]
        
        probs_a = predict_audio_probs(audio_model, audio_classes, pereche["audio_path"])
        probs_i = predict_image_probs(image_model, image_classes, pereche["image_path"])
        
        pred_a = audio_classes[np.argmax(probs_a)]
        pred_i = image_classes[np.argmax(probs_i)]
        pred_f = late_fusion(probs_a, audio_classes, probs_i, image_classes, alpha=0.5)
        
        y_true.append(adevarat)
        y_pred_audio.append(pred_a)
        y_pred_image.append(pred_i)
        y_pred_fusion.append(pred_f)
        
    def raport(nume, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division= 0)
        print(f"\n{nume}")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
        return acc, prec, rec, f1

    print(f"Evaluare pe {len(perechi)} perechi audio-imagine\n")
    raport("AUDIO (pe perechi)", y_true, y_pred_audio)
    raport("IMAGINE (pe perechi)", y_true, y_pred_image)
    raport("LATE FUSION", y_true, y_pred_fusion)
    
if __name__ == "__main__":
    evaluate()

    
        