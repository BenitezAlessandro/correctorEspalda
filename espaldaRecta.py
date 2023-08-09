import cv2
import mediapipe as mp
import pygame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pygame.init()

pygame.mixer.music.load('Concha_Tu_Madre.mp3')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

shoulder_threshold = 350

audio_playing = False  # Variable para controlar la reproducción del audio

with mp_pose.Pose(
    static_image_mode=False) as pose:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks is not None:
            left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
            right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
            
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
            cv2.line(frame, (0, shoulder_threshold), (width, shoulder_threshold), (0, 255, 0), 2)
            
            if left_shoulder_y > shoulder_threshold and right_shoulder_y > shoulder_threshold:
                if not audio_playing:  # Si el audio no está reproduciéndose
                    pygame.mixer.music.play(-1)  # Reproduce en bucle
                    audio_playing = True
            else:
                if audio_playing:  # Si el audio está reproduciéndose
                    pygame.mixer.music.stop()
                    audio_playing = False
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

pygame.quit()

cap.release()
cv2.destroyAllWindows()
