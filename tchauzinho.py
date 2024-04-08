import cv2
import mediapipe as mp
import math
from time import time

turns = 0  #int variable --> saves the number of times the wrist-motion way changes
old_ang = None #int variable --> saves the hand wrist-motion angle from the last loop
motion = None #boolean variable --> 1 == increasing angle | 2 == decreasing angle
old_motion = None #boolean variable --> saves the wrist-motion way from the last loop
start_time = None

# Inicializar o modelo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils  # Importar a função draw_landmarks



# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)  # Use 0 para a webcam padrão

def is_hand_open(hand_landmarks):
    if not hand_landmarks:
        return False
    
    # Define the landmark indices for fingertips and their corresponding base landmarks
    finger_tip_indices = [mp.solutions.hands.HandLandmark.THUMB_TIP,
                            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                            mp.solutions.hands.HandLandmark.PINKY_TIP]

    finger_base_indices = [mp.solutions.hands.HandLandmark.THUMB_MCP,
                            mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP,
                            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
                            mp.solutions.hands.HandLandmark.RING_FINGER_MCP,
                            mp.solutions.hands.HandLandmark.PINKY_MCP]

    # Check if all finger tips are above their corresponding base landmarks
    all_fingers_open = True
    for tip_idx, base_idx in zip(finger_tip_indices, finger_base_indices):
        tip_point = hand_landmarks.landmark[tip_idx]
        base_point = hand_landmarks.landmark[base_idx]

        # Check if the tip point is above the base point (in the y-axis)
        if tip_point.y >= base_point.y:
            all_fingers_open = False
            break

    return all_fingers_open

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame de BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar o frame com o modelo Hands
    results = hands.process(rgb_frame)

    # Verificar se mãos são detectadas
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[-1]
        if is_hand_open(hand_landmarks):

            if start_time == None:
                start_time = time()

            # Extrair coordenadas dos pontos-chave da mão
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            #get writs and middle_finger_tip positions
            height, width, _ = frame.shape  # Get frame dimensions
            wrist_x, wrist_y = int( wrist.x * width ), int(wrist.y * height )
            middle_X, middle_y = int( middle_finger_tip.x * width ), int( middle_finger_tip.y * height )


            #calculate wrist-motion angle
            if middle_X == wrist_x:
                hand_angulation = 90
            else:
                hand_angulation = int( math.degrees( math.atan( ( wrist_y - middle_y ) / ( middle_X - wrist_x ) ) ) / 25 )


            #checking for wrist-motion turns
            if old_ang != None:
                if hand_angulation > old_ang:
                    motion = 1
                elif hand_angulation < old_ang:
                    motion = 0
            
            if old_motion != None:
                if old_motion != motion:
                    turns += 1
                    start_time = time()
            
            old_motion = motion
            old_ang = hand_angulation

            if time() - start_time >= 1:
                old_motion = None
                old_ang = None
                turns = 0
                start_time = None

            if turns >= 4:
                cv2.putText(frame, f'tchauzinho detected!!', (int(width/2), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'turns {turns} | angulo: {hand_angulation*25}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            old_motion = None
            old_ang = None
            turns = 0
            start_time = None

        # Desenhar landmarks da mão no frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Hands: {len(results.multi_hand_landmarks)}', (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir o frame resultante
    cv2.imshow('Tchauzinho', frame)

    # Parar o loop quando pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
