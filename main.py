"""
Push-Up Coach AI - Système d'analyse de pompes en temps réel
Intègre la détection de pose 2D/3D avec analyse de la forme
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
import time
import json
from datetime import datetime
from pushup_analyzer import PushUpAnalyzer

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
            color = (0, 255, 0)  # Vert par défaut
            
            # Couleurs spéciales pour certains points
            if i in [5, 6]:  # Épaules
                color = (255, 165, 0)  # Orange
            elif i in [7, 8]:  # Coudes
                color = (255, 0, 0)  # Rouge
            elif i in [11, 12]:  # Hanches
                color = (0, 255, 255)  # Cyan
            
            cv2.circle(frame, (int(x * w), int(y * h)), 6, color, -1)
            cv2.circle(frame, (int(x * w), int(y * h)), 8, (255, 255, 255), 1)
            
            # Numéro du point (optionnel - désactiver pour plus de clarté)
            # cv2.putText(frame, str(i+1), (int(x * w) + 10, int(y * h)), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Dessiner les connexions
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        
        if (idx1 < len(keypoints) and idx2 < len(keypoints) and
            keypoints[idx1, 2] > confidence_threshold and 
            keypoints[idx2, 2] > confidence_threshold):
            
            x1, y1 = int(keypoints[idx1, 0] * w), int(keypoints[idx1, 1] * h)
            x2, y2 = int(keypoints[idx2, 0] * w), int(keypoints[idx2, 1] * h)
            
            # Épaisseur variable selon la connexion
            thickness = 3
            if connection in [(6, 8), (7, 9), (8, 10), (9, 11)]:  # Bras
                thickness = 4
            elif connection in [(12, 13), (6, 12), (7, 13)]:  # Tronc
                thickness = 4
            
            cv2.line(frame, (x1, y1), (x2, y2), (0, 100, 255), thickness)
    
    return frame

def visualize_3d_pose(keypoints_3d, save_path="pose_3d_reconstruction.png", title="Reconstruction 3D"):
    """Visualise le squelette 3D"""
    if len(keypoints_3d) == 0:
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Points
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], 
               c='r', marker='o', s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Connexions
    for connection in SKELETON_CONNECTIONS:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        
        if idx1 < len(keypoints_3d) and idx2 < len(keypoints_3d):
            x = [keypoints_3d[idx1, 0], keypoints_3d[idx2, 0]]
            y = [keypoints_3d[idx1, 1], keypoints_3d[idx2, 1]]
            z = [keypoints_3d[idx1, 2], keypoints_3d[idx2, 2]]
            
            # Couleur différente pour les bras/jambes
            color = 'blue'
            linewidth = 2
            if connection in [(6, 8), (7, 9), (8, 10), (9, 11)]:  # Bras
                color = 'red'
                linewidth = 3
            elif connection in [(16, 14), (14, 12), (17, 15), (15, 13)]:  # Jambes
                color = 'green'
                linewidth = 3
            
            ax.plot(x, y, z, c=color, linewidth=linewidth, alpha=0.8)
    
    # Configuration des axes
    ax.set_xlabel('X (Largeur)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Hauteur)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (Profondeur)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Limites des axes
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([80, 120])
    
    # Grille et perspective
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    
    # Ajouter une légende simple
    ax.text2D(0.05, 0.95, "● Points articulaires\n─ Connexions", 
              transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[3D] Visualisation sauvegardée : {save_path}")

# --- FONCTIONS D'INTERFACE PUSH-UP ---
def draw_pushup_interface(frame, analysis_results, rep_count, session_time, rep_detected=False):
    """Dessine l'interface utilisateur pour les pompes"""
    h, w = frame.shape[:2]
    
    # --- Overlay supérieur (informations) ---
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay_top, 0.7, frame, 0.3, 0, frame)
    
    # --- Overlay inférieur (feedback) ---
    if analysis_results['feedback']:
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h-80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bottom, 0.7, frame, 0.3, 0, frame)
    
    # Titre principal
    cv2.putText(frame, "🏋️ PUSH-UP COACH AI", (10, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 200, 255), 2)
    
    # --- Compteur de pompes (CENTRAL) ---
    counter_bg = frame.copy()
    cv2.rectangle(counter_bg, (w//2 - 80, 60), (w//2 + 80, 120), (30, 30, 30), -1)
    cv2.addWeighted(counter_bg, 0.6, frame, 0.4, 0, frame)
    
    # Bordure animée lors d'une répétition
    border_color = (0, 255, 0) if rep_detected else (100, 100, 100)
    border_thickness = 3 if rep_detected else 1
    cv2.rectangle(frame, (w//2 - 80, 60), (w//2 + 80, 120), border_color, border_thickness)
    
    # Nombre de pompes (grand)
    cv2.putText(frame, f"{rep_count}", (w//2 - 40, 105), 
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 0), 4)
    cv2.putText(frame, "POMPES", (w//2 - 50, 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # --- Panneau gauche (score et état) ---
    # Score de forme
    score = analysis_results['form_score']
    score_color = (0, 255, 0) if score > 80 else \
                 (0, 165, 255) if score > 60 else \
                 (0, 0, 255)
    
    # Cercle de score
    center_x, center_y = 70, 85
    radius = 25
    cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), -1)
    cv2.circle(frame, (center_x, center_y), radius, score_color, 2)
    
    # Pourcentage dans le cercle
    cv2.putText(frame, f"{score}", (center_x - 20, center_y + 10), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, score_color, 2)
    
    cv2.putText(frame, "SCORE", (center_x - 25, center_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # État actuel (UP/DOWN)
    state = analysis_results['state']
    state_color = (0, 255, 0) if state == "UP" else (0, 0, 255)
    cv2.putText(frame, f"ETAT: {state}", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    
    # --- Panneau droit (statistiques) ---
    stats_x = w - 180
    
    # Temps de session
    cv2.putText(frame, f"⏱️ {int(session_time)}s", 
                (stats_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Angles (si disponibles)
    y_offset = 110
    for angle_name, angle_value in analysis_results.get('angles', {}).items():
        if "elbow" in angle_name or "hip" in angle_name:
            display_name = "Coude" if "elbow" in angle_name else "Hanches"
            cv2.putText(frame, f"{display_name}: {angle_value:.0f}°", 
                       (stats_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
            y_offset += 20
    
    # --- Barre de progression d'amplitude ---
    if 'elbow_left' in analysis_results.get('angles', {}) or 'elbow_right' in analysis_results.get('angles', {}):
        # Calculer la progression
        elbow_angles = []
        if 'elbow_left' in analysis_results['angles']:
            elbow_angles.append(analysis_results['angles']['elbow_left'])
        if 'elbow_right' in analysis_results['angles']:
            elbow_angles.append(analysis_results['angles']['elbow_right'])
        
        if elbow_angles:
            avg_angle = np.mean(elbow_angles)
            # Progression: 0% quand coude à 180° (bras tendu), 100% quand coude à 90° (pliure complète)
            progress = 1.0 - min(max(avg_angle - 90, 0) / 90, 1.0)
            
            # Position et dimensions
            bar_x, bar_y = 20, h - 40
            bar_width, bar_height = w - 40, 15
            
            # Barre de fond
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Barre de progression
            progress_width = int(bar_width * progress)
            bar_color = (0, 255, 0) if progress > 0.7 else \
                       (0, 165, 255) if progress > 0.4 else \
                       (0, 0, 255)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                         bar_color, -1)
            
            # Contour
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (200, 200, 200), 1)
            
            # Texte
            cv2.putText(frame, "Amplitude", (bar_x, bar_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"{int(progress*100)}%", 
                       (bar_x + bar_width + 5, bar_y + bar_height//2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)
    
    # --- Feedback textuel ---
    if analysis_results['feedback']:
        fb = analysis_results['feedback']
        # Déterminer la couleur en fonction du type de feedback
        if "✅" in fb:
            color = (0, 255, 0)  # Vert
            emoji = "✅"
        elif "🔴" in fb:
            color = (0, 0, 255)  # Rouge
            emoji = "🔴"
        elif "🟡" in fb:
            color = (0, 165, 255)  # Orange
            emoji = "🟡"
        else:
            color = (255, 255, 255)  # Blanc
            emoji = ""
        
        # Afficher le feedback (sans l'emoji dans le texte)
        fb_text = fb.replace("✅", "").replace("🔴", "").replace("🟡", "").strip()
        if emoji:
            # Afficher l'emoji séparément
            cv2.putText(frame, emoji, (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, fb_text, (50, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, fb_text, (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # --- Indicateur de position de caméra ---
    if analysis_results.get('person_detected', True) == False:
        cv2.putText(frame, "↻ Tournez-vous face à la caméra", 
                   (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

def save_session_data(pushup_analyzer, filename="session_history.json"):
    """Sauvegarde les données de la session"""
    summary = pushup_analyzer.get_session_summary()
    
    session_data = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_reps': summary['total_reps'],
        'session_duration': summary['session_duration'],
        'avg_rep_time': float(summary['avg_rep_time']),
        'total_errors': summary['total_errors'],
        'common_errors': summary['common_errors'],
        'rep_times': [float(t) for t in pushup_analyzer.rep_times]
    }
    
    try:
        # Charger l'historique existant
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # Ajouter la nouvelle session
        history.append(session_data)
        
        # Sauvegarder
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"[SAVE] Données sauvegardées dans {filename}")
        return True
    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder: {e}")
        return False

# --- FONCTION PRINCIPALE ---
def main():
    # 1. Charger le modèle YOLOv8-Pose
    print("="*60)
    print("        PUSH-UP COACH AI - Initialisation")
    print("="*60)
    print("[1/3] Chargement du modèle YOLOv8-Pose...")
    try:
        model = YOLO('yolov8n-pose.pt')  # Téléchargement automatique si besoin
        print("   ✓ Modèle chargé avec succès")
    except Exception as e:
        print(f"   ✗ Erreur lors du chargement du modèle: {e}")
        print("   Essayez: pip install ultralytics")
        return
    
    # 2. Initialiser l'analyseur de pompes
    print("[2/3] Initialisation de l'analyseur de pompes...")
    try:
        pushup_analyzer = PushUpAnalyzer()
        print("   ✓ Analyseur initialisé")
    except Exception as e:
        print(f"   ✗ Erreur lors de l'initialisation: {e}")
        return
    
    # 3. Ouvrir la caméra
    print("[3/3] Ouverture de la caméra...")
    cap = cv2.VideoCapture(0)
    
    # Configuration caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("   ✗ Erreur: Impossible d'ouvrir la caméra")
        print("   Vérifiez que la caméra est connectée et non utilisée par une autre application")
        return
    
    print("   ✓ Caméra ouverte (1280x720 @ 30 FPS)")
    
    # Afficher les instructions
    print("\n" + "="*60)
    print("          🏋️ PRÊT POUR L'ENTRAÎNEMENT !")
    print("="*60)
    print("Commandes:")
    print("  [Q] : Quitter l'application")
    print("  [S] : Sauvegarder la pose 3D actuelle")
    print("  [R] : Réinitialiser le compteur")
    print("  [P] : Afficher les statistiques en temps réel")
    print("  [D] : Sauvegarder les données de session")
    print("  [C] : Changer de caméra (si disponible)")
    print("="*60)
    print("Positionnez-vous face à la caméra, les épaules visibles")
    print("Commencez vos pompes ! Le système comptera automatiquement")
    print("="*60 + "\n")
    
    # Variables d'état
    last_keypoints_3d = None
    session_start_time = time.time()
    last_rep_time = time.time()
    camera_index = 0
    show_angles = False
    session_saved = False
    
    # Variables pour l'animation
    rep_animation_frames = 0
    max_animation_frames = 10
    
    # Boucle principale
    while True:
        # Lire une frame
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Impossible de lire depuis la caméra")
            time.sleep(1)
            continue
        
        # Redimensionner pour le traitement
        frame_resized = cv2.resize(frame, (640, 480))
        frame_display = frame_resized.copy()
        
        # 3. Détection de pose avec YOLOv8
        try:
            results = model(frame_resized, conf=0.5, verbose=False)
        except Exception as e:
            print(f"[ERREUR] Détection de pose échouée: {e}")
            continue
        
        person_detected = len(results[0].keypoints) > 0
        
        if person_detected:
            # Extraire les keypoints (format normalisé 0-1)
            keypoints_data = results[0].keypoints.data[0].cpu().numpy()
            
            # Dessiner le squelette 2D
            frame_with_skeleton = draw_skeleton_2d(frame_resized.copy(), keypoints_data)
            frame_display = frame_with_skeleton
            
            # Analyser les pompes
            analysis_results = pushup_analyzer.analyze_frame(keypoints_data, frame_resized.shape)
            analysis_results['person_detected'] = True
            
            # Reconstruction 3D (pour sauvegarde)
            keypoints_3d = lift_2d_to_3d(keypoints_data)
            last_keypoints_3d = keypoints_3d
            
            # Vérifier si une répétition vient d'être détectée
            rep_detected_now = analysis_results['rep_detected']
            if rep_detected_now:
                rep_animation_frames = max_animation_frames
                last_rep_time = time.time()
                print(f"[POMPE #{pushup_analyzer.rep_count}] Score: {analysis_results['form_score']}/100 - {analysis_results['feedback']}")
            
            # Mettre à jour l'animation
            if rep_animation_frames > 0:
                rep_animation_frames -= 1
            
            # Dessiner l'interface
            session_time = time.time() - session_start_time
            frame_display = draw_pushup_interface(
                frame_display,
                analysis_results,
                pushup_analyzer.rep_count,
                session_time,
                rep_detected=(rep_animation_frames > 0)
            )
            
            # Afficher les angles en temps réel (optionnel)
            if show_angles and 'angles' in analysis_results:
                y_pos = 180
                for angle_name, angle_value in analysis_results['angles'].items():
                    cv2.putText(frame_display, f"{angle_name}: {angle_value:.1f}°", 
                               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
                    y_pos += 20
            
            # Indicateur de dernière répétition
            time_since_last_rep = time.time() - last_rep_time
            if time_since_last_rep < 2.0 and pushup_analyzer.rep_count > 0:
                cv2.putText(frame_display, f"Dernière: {time_since_last_rep:.1f}s", 
                           (frame_display.shape[1] - 150, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
            
        else:
            # Aucune personne détectée
            frame_display = frame_resized
            cv2.putText(frame_display, "AUCUNE PERSONNE DETECTEE", 
                       (frame_display.shape[1]//2 - 150, frame_display.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame_display, "Positionnez-vous face a la camera", 
                       (frame_display.shape[1]//2 - 180, frame_display.shape[0]//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Afficher la frame
        cv2.imshow('🏋️ Push-Up Coach AI', frame_display)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' ou Échap
            print("\n" + "="*60)
            print("            FIN DE LA SESSION")
            print("="*60)
            break
            
        elif key == ord('s') and last_keypoints_3d is not None:
            # Sauvegarder la pose 3D
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_3d/pushup_{pushup_analyzer.rep_count}reps_{timestamp}.png"
            
            # Créer le dossier si nécessaire
            import os
            os.makedirs("pose_3d", exist_ok=True)
            
            visualize_3d_pose(
                last_keypoints_3d, 
                filename,
                f"Pompe #{pushup_analyzer.rep_count} - {timestamp}"
            )
            
        elif key == ord('r'):
            # Réinitialiser le compteur
            pushup_analyzer.reset_session()
            session_start_time = time.time()
            last_rep_time = time.time()
            print("[RESET] Compteur réinitialisé !")
            
        elif key == ord('p'):
            # Afficher les statistiques
            summary = pushup_analyzer.get_session_summary()
            print("\n" + "="*50)
            print("STATISTIQUES EN TEMPS REEL")
            print("="*50)
            print(f"Pompes complétées: {summary['total_reps']}")
            print(f"Durée de session: {summary['session_duration']:.1f}s")
            print(f"Temps moyen par pompe: {summary['avg_rep_time']:.2f}s")
            print(f"Erreurs détectées: {summary['total_errors']}")
            
            if summary['common_errors']:
                print("Erreurs fréquentes:")
                for error in summary['common_errors'][:3]:
                    print(f"  • {error}")
            print("="*50)
            
        elif key == ord('d'):
            # Sauvegarder les données
            if save_session_data(pushup_analyzer):
                session_saved = True
                print("[INFO] Données sauvegardées avec succès")
            else:
                print("[ERREUR] Échec de la sauvegarde")
                
        elif key == ord('c'):
            # Changer de caméra
            camera_index += 1
            cap.release()
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"[CAMERA] Changé vers la caméra #{camera_index}")
            else:
                print(f"[CAMERA] Caméra #{camera_index} non disponible")
                camera_index = 0
                cap = cv2.VideoCapture(camera_index)
                
        elif key == ord('a'):
            # Basculer l'affichage des angles
            show_angles = not show_angles
            print(f"[DEBUG] Affichage des angles: {'ACTIVÉ' if show_angles else 'DÉSACTIVÉ'}")
            
        elif key == ord(' '):
            # Pause
            print("[PAUSE] Appuyez sur une touche pour continuer...")
            cv2.waitKey(0)
    
    # --- FIN DE LA SESSION ---
    
    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    
    # Afficher le résumé final
    summary = pushup_analyzer.get_session_summary()
    
    print("\n" + "="*60)
    print("            RÉSUMÉ FINAL DE LA SESSION")
    print("="*60)
    print(f"🏆 TOTAL DE POMPES: {summary['total_reps']}")
    print(f"⏱️  Durée totale: {summary['session_duration']:.1f} secondes")
    
    if summary['total_reps'] > 0:
        print(f"📊 Temps moyen par pompe: {summary['avg_rep_time']:.2f}s")
        reps_per_minute = (summary['total_reps'] / summary['session_duration']) * 60
        print(f"⚡ Vitesse: {reps_per_minute:.1f} pompes/minute")
    
    print(f"🎯 Erreurs détectées: {summary['total_errors']}")
    
    if summary['common_errors']:
        print("\n📋 PRINCIPALES CORRECTIONS À APPORTER:")
        for i, error in enumerate(summary['common_errors'][:3], 1):
            print(f"  {i}. {error}")
    
    print("\n💪 CONSEILS POUR LA PROCHAINE SESSION:")
    if summary['total_reps'] == 0:
        print("  • Assurez-vous que votre corps entier est visible par la caméra")
        print("  • Effectuez des mouvements complets (descente jusqu'à 90°)")
    elif summary['avg_rep_time'] < 1.0:
        print("  • Ralentissez le mouvement pour mieux contrôler la forme")
        print("  • Concentrez-vous sur la qualité plutôt que la quantité")
    else:
        print("  • Continuez ainsi ! Votre rythme est excellent")
        print("  • Essayez d'augmenter légèrement le nombre de répétitions")
    
    # Proposer la sauvegarde si pas déjà fait
    if not session_saved and summary['total_reps'] > 0:
        print("\n💾 Voulez-vous sauvegarder les données de cette session? (O/N)")
        # Dans une vraie application, vous pourriez ajouter une interface graphique
        # Pour cette version console, on sauvegarde automatiquement
        save_session_data(pushup_analyzer)
    
    print("\n" + "="*60)
    print("  Merci d'avoir utilisé Push-Up Coach AI !")
    print("  À bientôt pour votre prochain entraînement 💪")
    print("="*60)

# --- POINT D'ENTRÉE ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Programme interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # S'assurer que toutes les fenêtres sont fermées
        cv2.destroyAllWindows()
        print("\n[INFO] Nettoyage effectué. Programme terminé.")