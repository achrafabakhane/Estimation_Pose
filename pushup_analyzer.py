"""
Analyseur de pompes - Module indépendant
"""

import numpy as np
import time
from scipy.spatial.distance import euclidean
import cv2

class PushUpAnalyzer:
    def __init__(self):
        # États possibles
        self.STATE_UP = "UP"
        self.STATE_DOWN = "DOWN"
        self.STATE_TRANSITION = "TRANSITION"
        
        # Configuration
        self.current_state = self.STATE_UP
        self.rep_count = 0
        self.last_rep_time = None
        self.rep_times = []
        self.form_errors_history = []
        self.session_start_time = time.time()
        
        # Seuils (ajustables)
        self.ELBOW_ANGLE_THRESHOLD = 90      # degrés pour considérer "bas"
        self.HIP_ANGLE_MIN = 160             # alignement dos minimum
        self.HIP_ANGLE_MAX = 185             # alignement dos maximum
        
        # Configuration visuelle
        self.colors = {
            'good': (0, 255, 0),      # Vert
            'warning': (0, 165, 255), # Orange
            'bad': (0, 0, 255),       # Rouge
            'text': (255, 255, 255)   # Blanc
        }
    
    def calculate_angle(self, a, b, c):
        """Calcule l'angle entre trois points (b comme sommet)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def calculate_distance(self, point1, point2):
        """Calcule la distance euclidienne entre deux points"""
        return euclidean(point1, point2)
    
    def analyze_frame(self, keypoints_2d, frame_shape):
        """
        Analyse une frame pour détecter et évaluer les pompes
        
        Args:
            keypoints_2d: Array des keypoints (17, 3) [x, y, confidence]
            frame_shape: Tuple (height, width) de l'image
        
        Returns:
            dict: Résultats de l'analyse
        """
        h, w = frame_shape[:2] if len(frame_shape) > 2 else frame_shape
        
        results = {
            'rep_detected': False,
            'form_score': 100,
            'errors': [],
            'angles': {},
            'feedback': "",
            'state': self.current_state,
            'count': self.rep_count,
            'visual_overlays': []  # Pour dessiner des éléments visuels
        }
        
        try:
            # Convertir les keypoints en coordonnées pixels
            keypoints_px = keypoints_2d.copy()
            keypoints_px[:, 0] *= w  # X
            keypoints_px[:, 1] *= h  # Y
            
            # Calcul des angles
            angles = self._calculate_angles(keypoints_px)
            results['angles'] = angles
            
            # Détection de pompe
            if 'elbow_left' in angles or 'elbow_right' in angles:
                # Utiliser la moyenne des deux coudes si disponibles
                elbow_angles = []
                if 'elbow_left' in angles and angles['elbow_left'] > 0:
                    elbow_angles.append(angles['elbow_left'])
                if 'elbow_right' in angles and angles['elbow_right'] > 0:
                    elbow_angles.append(angles['elbow_right'])
                
                if elbow_angles:
                    avg_elbow_angle = np.mean(elbow_angles)
                    
                    # Logique de détection
                    rep_detected = self._detect_rep(avg_elbow_angle)
                    if rep_detected:
                        results['rep_detected'] = True
            
            # Analyse de la forme
            form_results = self._analyze_form(angles, keypoints_px)
            results['form_score'] = form_results['score']
            results['errors'] = form_results['errors']
            results['feedback'] = self._generate_feedback(form_results['errors'])
            
            # Éléments visuels à dessiner
            results['visual_overlays'] = self._generate_visual_overlays(
                angles, keypoints_px, results
            )
            
        except Exception as e:
            print(f"Erreur dans l'analyse: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _calculate_angles(self, keypoints_px):
        """Calcule tous les angles importants"""
        angles = {}
        
        # Indices selon votre configuration KEYPOINTS
        # 0: Nez, 5: Épaule gauche, 6: Épaule droite, 7: Coude gauche, 8: Coude droit
        # 9: Poignet gauche, 10: Poignet droit, 11: Hanche gauche, 12: Hanche droite
        # 13: Genou gauche, 14: Genou droit
        
        # Angle coude gauche
        if all(keypoints_px[i, 2] > 0.5 for i in [5, 7, 9]):  # Confiance > 50%
            angles['elbow_left'] = self.calculate_angle(
                keypoints_px[5, :2],  # Épaule gauche
                keypoints_px[7, :2],  # Coude gauche
                keypoints_px[9, :2]   # Poignet gauche
            )
        
        # Angle coude droit
        if all(keypoints_px[i, 2] > 0.5 for i in [6, 8, 10]):
            angles['elbow_right'] = self.calculate_angle(
                keypoints_px[6, :2],  # Épaule droite
                keypoints_px[8, :2],  # Coude droit
                keypoints_px[10, :2]  # Poignet droit
            )
        
        # Angle hanche (alignement du dos)
        if all(keypoints_px[i, 2] > 0.5 for i in [5, 11, 13]):
            angles['hip_left'] = self.calculate_angle(
                keypoints_px[5, :2],   # Épaule gauche
                keypoints_px[11, :2],  # Hanche gauche
                keypoints_px[13, :2]   # Genou gauche
            )
        
        if all(keypoints_px[i, 2] > 0.5 for i in [6, 12, 14]):
            angles['hip_right'] = self.calculate_angle(
                keypoints_px[6, :2],   # Épaule droite
                keypoints_px[12, :2],  # Hanche droite
                keypoints_px[14, :2]   # Genou droite
            )
        
        return angles
    
    def _detect_rep(self, elbow_angle):
        """Détecte une répétition complète"""
        rep_detected = False
        
        if self.current_state == self.STATE_UP and elbow_angle < self.ELBOW_ANGLE_THRESHOLD:
            self.current_state = self.STATE_DOWN
            
        elif self.current_state == self.STATE_DOWN and elbow_angle > 160:
            self.current_state = self.STATE_UP
            self.rep_count += 1
            rep_detected = True
            
            # Enregistrer le temps
            current_time = time.time()
            if self.last_rep_time:
                rep_duration = current_time - self.last_rep_time
                self.rep_times.append(rep_duration)
            self.last_rep_time = current_time
        
        return rep_detected
    
    def _analyze_form(self, angles, keypoints_px):
        """Analyse la qualité de la forme"""
        score = 100
        errors = []
        
        # 1. Vérifier l'alignement du dos
        hip_angles = []
        if 'hip_left' in angles:
            hip_angles.append(angles['hip_left'])
        if 'hip_right' in angles:
            hip_angles.append(angles['hip_right'])
        
        if hip_angles:
            avg_hip_angle = np.mean(hip_angles)
            
            if avg_hip_angle < self.HIP_ANGLE_MIN:
                errors.append("Dos cambré")
                score -= 25
            elif avg_hip_angle > self.HIP_ANGLE_MAX:
                errors.append("Hanches trop hautes")
                score -= 20
        
        # 2. Vérifier l'amplitude
        elbow_angles = []
        if 'elbow_left' in angles:
            elbow_angles.append(angles['elbow_left'])
        if 'elbow_right' in angles:
            elbow_angles.append(angles['elbow_right'])
        
        if elbow_angles:
            avg_elbow = np.mean(elbow_angles)
            
            if self.current_state == self.STATE_DOWN and avg_elbow > 100:
                errors.append("Amplitude insuffisante")
                score -= 30
            elif avg_elbow < 70:
                errors.append("Trop bas - risque de blessure")
                score -= 15
        
        # 3. Vérifier la symétrie
        if 'elbow_left' in angles and 'elbow_right' in angles:
            angle_diff = abs(angles['elbow_left'] - angles['elbow_right'])
            if angle_diff > 20:
                errors.append("Mouvement asymétrique")
                score -= 10
        
        score = max(0, score)  # Pas de score négatif
        
        return {'score': score, 'errors': errors}
    
    def _generate_feedback(self, errors):
        """Génère un feedback textuel"""
        if not errors:
            return "✅ Forme parfaite !"
        
        # Prioriser les erreurs
        error_priority = {
            "Dos cambré": "🔴 CORRIGEZ: Gardez le dos droit !",
            "Amplitude insuffisante": "🟡 DESCENDEZ plus bas !",
            "Hanches trop hautes": "🟡 Baissez les hanches !",
            "Trop bas - risque de blessure": "🔴 ATTENTION: Trop bas !",
            "Mouvement asymétrique": "🟡 Travaillez la symétrie"
        }
        
        for error in errors:
            if error in error_priority:
                return error_priority[error]
        
        return "🟡 Ajustez votre position"
    
    def _generate_visual_overlays(self, angles, keypoints_px, results):
        """Génère les éléments visuels à dessiner"""
        overlays = []
        
        # Barre de progression
        if 'elbow_left' in angles or 'elbow_right' in angles:
            elbow_angles = []
            if 'elbow_left' in angles:
                elbow_angles.append(angles['elbow_left'])
            if 'elbow_right' in angles:
                elbow_angles.append(angles['elbow_right'])
            
            if elbow_angles:
                avg_angle = np.mean(elbow_angles)
                progress = 1.0 - min(avg_angle / 180, 1.0)
                
                overlays.append({
                    'type': 'progress_bar',
                    'position': (50, 30, 200, 20),
                    'progress': progress,
                    'color': self.colors['good'] if progress > 0.5 else self.colors['warning']
                })
        
        # Indicateur d'état
        state_color = self.colors['good'] if self.current_state == "UP" else self.colors['bad']
        overlays.append({
            'type': 'text',
            'text': f"ÉTAT: {self.current_state}",
            'position': (50, 60),
            'color': state_color
        })
        
        return overlays
    
    def get_session_summary(self):
        """Retourne un résumé de la session"""
        session_duration = time.time() - self.session_start_time
        
        summary = {
            'total_reps': self.rep_count,
            'session_duration': session_duration,
            'avg_rep_time': np.mean(self.rep_times) if self.rep_times else 0,
            'total_errors': len(self.form_errors_history),
            'common_errors': list(set([e for sublist in self.form_errors_history for e in sublist]))[:5]
        }
        
        return summary
    
    def reset_session(self):
        """Réinitialise la session actuelle"""
        self.current_state = self.STATE_UP
        self.rep_count = 0
        self.last_rep_time = None
        self.rep_times = []
        self.form_errors_history = []
        self.session_start_time = time.time()