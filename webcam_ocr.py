import cv2
import easyocr

reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)

WIDTH, HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao capturar o frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small_frame = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

    results = reader.readtext(small_frame)
    
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"Texto detectado: {text} (confiança: {prob:.2f})")

    cv2.imshow('OCR Results', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()