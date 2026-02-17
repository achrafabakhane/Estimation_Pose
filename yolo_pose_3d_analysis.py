import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO

# --- CONFIGURATION ---
KEYPOINTS = [
    "Nez", "Œil Gauche", "Œil Droit", "Oreille Gauche", "Oreille Droite",
    "Épaule Gauche", "Épaule Droite", "Coude Gauche", "Coude Droit",
    "Poignet Gauche", "Poignet Droit", "Hanche Gauche", "Hanche Droite",
    "Genou Gauche", "Genou Droit", "Cheville Gauche", "Cheville Droite"
]

SKELETON_CONNECTIONS = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7)
]

# --- FONCTIONS DE RECONSTRUCTION 3D ---
def lift_2d_to_3d(keypoints_2d):
    """Convertit les points 2D en 3D"""
    if len(keypoints_2d) == 0:
        return np.array([])
    
    keypoints_3d = np.zeros((len(keypoints_2d), 3))
    keypoints_3d[:, 0] = keypoints_2d[:, 0]  # X
    keypoints_3d[:, 1] = keypoints_2d[:, 1]  # Y
    
    # Estimation de la profondeur (simplifiée)
    if len(keypoints_2d) >= 13:  # Au moins les hanches
        # Utiliser la largeur des épaules pour estimer la profondeur
        if keypoints_2d[5, 2] > 0.5 and keypoints_2d[6, 2] > 0.5:  # Confiance > 50%
            shoulder_width = abs(keypoints_2d[6, 0] - keypoints_2d[5, 0])
            depth_base = 100 - (shoulder_width / 10)  # Plus les épaules sont larges, plus proche
            
            # Assigner des profondeurs différentes selon les parties du corps
            keypoints_3d[0:5, 2] = depth_base + 5  # Tête
            keypoints_3d[5:7, 2] = depth_base  # Épaules
            keypoints_3d[11:13, 2] = depth_base - 5  # Hanches
            
            # Bras et jambes
            for i in range(len(keypoints_2d)):
                if i >= 7 and i <= 10:  # Bras
                    keypoints_3d[i, 2] = depth_base + (10 if i % 2 == 0 else -10)
                elif i >= 13 and i <= 16:  # Jambes
                    keypoints_3d[i, 2] = depth_base - 10 + (5 if i % 2 == 0 else -5)
    
    return keypoints_3d

def draw_skeleton_2d(frame, keypoints, confidence_threshold=0.5):
    """Dessine le squelette sur l'image 2D"""
    h, w = frame.shape[:2]
    
    # Dessiner les points
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (int(x * w), int(y * h - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Dessiner les connexions
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        
        if (idx1 < len(keypoints) and idx2 < len(keypoints) and
            keypoints[idx1, 2] > confidence_threshold and 
            keypoints[idx2, 2] > confidence_threshold):
            
            x1, y1 = int(keypoints[idx1, 0] * w), int(keypoints[idx1, 1] * h)
            x2, y2 = int(keypoints[idx2, 0] * w), int(keypoints[idx2, 1] * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return frame

def visualize_3d_pose(keypoints_3d, save_path="pose_3d_reconstruction.png"):
    """Visualise le squelette 3D"""
    if len(keypoints_3d) == 0:
        return
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Points
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], 
               c='r', marker='o', s=50)
    
    # Connexions
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        
        if idx1 < len(keypoints_3d) and idx2 < len(keypoints_3d):
            x = [keypoints_3d[idx1, 0], keypoints_3d[idx2, 0]]
            y = [keypoints_3d[idx1, 1], keypoints_3d[idx2, 1]]
            z = [keypoints_3d[idx1, 2], keypoints_3d[idx2, 2]]
            
            ax.plot(x, y, z, c='b', linewidth=2)
    
    ax.set_xlabel('X (Largeur)')
    ax.set_ylabel('Y (Hauteur)')
    ax.set_zlabel('Z (Profondeur)')
    ax.set_title('Reconstruction 3D en Temps Réel')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Visualisation 3D sauvegardée : {save_path}")

# --- FONCTION PRINCIPALE ---
def main():
    # 1. Charger le modèle YOLOv8-Pose
    print("Chargement du modèle YOLOv8-Pose...")
    model = YOLO('yolov8n-pose.pt')  # Téléchargement automatique
    
    # 2. Ouvrir la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    print("Caméra ouverte. Appuyez sur 'q' pour quitter, 's' pour sauvegarder la pose 3D")
    
    last_keypoints_3d = None
    
    while True:
        # Lire une frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensionner pour meilleures performances
        frame_resized = cv2.resize(frame, (640, 480))
        
        # 3. Détection de pose avec YOLOv8
        results = model(frame_resized, conf=0.5)
        
        if len(results[0].keypoints) > 0:
            # Extraire les keypoints (format normalisé 0-1)
            keypoints_data = results[0].keypoints.data[0].cpu().numpy()
            
            # 4. Dessiner le squelette 2D
            frame_with_skeleton = draw_skeleton_2d(frame_resized.copy(), keypoints_data)
            
            # 5. Reconstruction 3D
            keypoints_3d = lift_2d_to_3d(keypoints_data)
            last_keypoints_3d = keypoints_3d
            
            # Afficher des informations
            cv2.putText(frame_with_skeleton, f"Personnes detectees: {len(results[0].boxes)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Afficher les coordonnées d'un point (ex: nez)
            if keypoints_data[0, 2] > 0.5:  # Nez avec bonne confiance
                cv2.putText(frame_with_skeleton, 
                          f"Nez: ({keypoints_data[0, 0]:.2f}, {keypoints_data[0, 1]:.2f})",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            frame_with_skeleton = frame_resized
            cv2.putText(frame_with_skeleton, "Aucune personne detectee", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Afficher la frame
        cv2.imshow('Pose Estimation - Camera', frame_with_skeleton)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and last_keypoints_3d is not None:
            visualize_3d_pose(last_keypoints_3d, f"pose_3d_{len(last_keypoints_3d)}.png")
            print("Pose 3D sauvegardee !")
        elif key == ord('r'):
            # Réinitialiser/recadrer
            print("Recadrage...")
    
    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    print("Programme termine.")

if __name__ == "__main__":
    main()
