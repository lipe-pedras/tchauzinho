import cv2
import mediapipe as mp
import math

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

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
        for hand_landmarks in results.multi_hand_landmarks:
            if is_hand_open(hand_landmarks):
                print("hand is open!")
                # Extrair coordenadas dos pontos-chave da mão
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                for landmark in [wrist, thumb_tip, index_finger_tip]:
                    # Get pixel coordinates of the landmark
                    height, width, _ = frame.shape  # Get frame dimensions
                    cx, cy = int(landmark.x * width), int(landmark.y * height)  # Convert normalized coordinates to pixel coordinates

                    # Print the pixel coordinates of specific landmarks
                    if landmark == hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                        print(f"Index Finger Tip - X: {cx}, Y: {cy}")
                    elif landmark == hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]:
                        print(f"Thumb Tip - X: {cx}, Y: {cy}")

                # Calcular distâncias relevantes
                dist_thumb_to_wrist = calculate_distance(thumb_tip, wrist)
                dist_index_to_wrist = calculate_distance(index_finger_tip, wrist)

                # Verificar se a mão está em posição de "tchauzinho"
                if dist_thumb_to_wrist < dist_index_to_wrist:
                    cv2.putText(frame, 'Tchauzinho detectado!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                else:
                    break

            # Desenhar landmarks da mão no frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    # Exibir o frame resultante
    cv2.imshow('Tchauzinho', frame)

    # Parar o loop quando pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
