import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class ReconhecimentoFacial:
    
    def __init__(self, pasta_dados="dados_faciais"):
        self.pasta_dados = Path(pasta_dados)
        self.pasta_dados.mkdir(exist_ok=True)
        
        self.arquivo_encodings = self.pasta_dados / "encodings.pkl"
        self.rostos_conhecidos = []
        self.nomes_conhecidos = []
        
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.carregar_encodings()
    
    def carregar_encodings(self):
        if self.arquivo_encodings.exists():
            with open(self.arquivo_encodings, 'rb') as f:
                dados = pickle.load(f)
                self.rostos_conhecidos = dados['encodings']
                self.nomes_conhecidos = dados['nomes']
            print(f"{len(self.nomes_conhecidos)} rostos carregados")
        else:
            print("Nenhum rosto cadastrado ainda")
    
    def salvar_encodings(self):
        dados = {
            'encodings': self.rostos_conhecidos,
            'nomes': self.nomes_conhecidos
        }
        with open(self.arquivo_encodings, 'wb') as f:
            pickle.dump(dados, f)
        print(f"Encodings salvos: {len(self.nomes_conhecidos)} rostos")
    
    def extrair_features_faciais(self, frame_rgb, face_locations):
        features_list = []
        
        for (x, y, w, h) in face_locations:
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame_rgb.shape[1], x + w)
            y2 = min(frame_rgb.shape[0], y + h)
            
            if x2 <= x or y2 <= y:
                continue
            
            face_roi = frame_rgb[y:y2, x:x2]
            
            if face_roi.size == 0:
                continue
            
            face_resized = cv2.resize(face_roi, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            face_hsv = cv2.cvtColor(face_resized, cv2.COLOR_RGB2HSV)
            
            hist_r = cv2.calcHist([face_resized], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([face_resized], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([face_resized], [2], None, [32], [0, 256])
            
            hist_h = cv2.calcHist([face_hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([face_hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([face_hsv], [2], None, [32], [0, 256])
            
            sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            
            hist_sobelx = np.histogram(sobelx, bins=32)[0]
            hist_sobely = np.histogram(sobely, bins=32)[0]
            
            face_tiny = cv2.resize(face_gray, (32, 32))
            pixels = face_tiny.flatten()
            
            features = np.concatenate([
                hist_r.flatten(),
                hist_g.flatten(),
                hist_b.flatten(),
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten(),
                hist_sobelx.flatten(),
                hist_sobely.flatten(),
                pixels.flatten()
            ])
            
            features = features.astype(np.float32)
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            features_list.append(features)
        
        return features_list
    
    def detectar_rostos(self, frame_rgb):
        results = self.face_detection.process(frame_rgb)
        face_locations = []
        
        if results.detections:
            h, w, _ = frame_rgb.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                face_locations.append((x, y, width, height))
        
        return face_locations
    
    def cadastrar_rosto(self, nome):
        print(f"\n=== Cadastrando: {nome} ===")
        print("Pressione ESPACO para capturar a foto")
        print("Pressione ESC para cancelar")
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Erro ao acessar a webcam")
            return False
        
        foto_capturada = False
        frame_capturado = None
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = self.detectar_rostos(rgb_frame)
            
            for (x, y, w, h) in face_locations:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.putText(frame, "ESPACO: Capturar | ESC: Cancelar", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(face_locations) > 0:
                cv2.putText(frame, f"Rosto detectado! ({nome})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Nenhum rosto detectado", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Cadastro de Rosto', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and len(face_locations) > 0:
                frame_capturado = rgb_frame
                foto_capturada = True
                print("Foto capturada!")
                break
            
            elif key == 27:
                print("Cadastro cancelado")
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        if foto_capturada:
            face_locations = self.detectar_rostos(frame_capturado)
            face_features = self.extrair_features_faciais(frame_capturado, face_locations)
            
            if len(face_features) > 0:
                self.rostos_conhecidos.append(face_features[0])
                self.nomes_conhecidos.append(nome)
                self.salvar_encodings()
                print(f"{nome} cadastrado com sucesso!")
                return True
            else:
                print("Nao foi possivel processar o rosto")
                return False
        
        return False
    
    def reconhecer_tempo_real(self):
        print("\n=== Reconhecimento Facial em Tempo Real ===")
        print("Pressione 'q' para sair")
        
        if len(self.rostos_conhecidos) == 0:
            print("Nenhum rosto cadastrado. Cadastre rostos primeiro!")
            return
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Erro ao acessar a webcam")
            return
        
        cached_face_data = []
        frame_count = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_count % 2 == 0:
                face_locations = self.detectar_rostos(rgb_frame)
                face_features = self.extrair_features_faciais(rgb_frame, face_locations)
                
                face_names = []
                for face_feature in face_features:
                    name = "Desconhecido"
                    confidence = 0
                    
                    if len(self.rostos_conhecidos) > 0:
                        similarities = []
                        for known_face in self.rostos_conhecidos:
                            sim = cosine_similarity(
                                face_feature.reshape(1, -1),
                                known_face.reshape(1, -1)
                            )[0][0]
                            similarities.append(sim)
                        
                        similarities = np.array(similarities)
                        best_match_index = np.argmax(similarities)
                        best_similarity = similarities[best_match_index]
                        
                        if len(similarities) > 1:
                            sorted_similarities = np.sort(similarities)[::-1]
                            second_best = sorted_similarities[1] if len(sorted_similarities) > 1 else 0
                            margin = best_similarity - second_best
                        else:
                            margin = 1.0
                        
                        if best_similarity > 0.45 and margin > 0.02:
                            name = self.nomes_conhecidos[best_match_index]
                            confidence = best_similarity * 100
                            name = f"{name} ({confidence:.1f}%)"
                    
                    face_names.append(name)
                
                cached_face_data = list(zip(face_locations, face_names))
            
            for (x, y, w, h), name in cached_face_data:
                color = (0, 255, 0) if "Desconhecido" not in name else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.rectangle(frame, (x, label_y - 25), (x + w, label_y), color, cv2.FILLED)
                cv2.putText(frame, name, (x + 6, label_y - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Rostos cadastrados: {len(self.nomes_conhecidos)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Pressione 'q' para sair", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Reconhecimento Facial - MediaPipe', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        video_capture.release()
        cv2.destroyAllWindows()
        print("Reconhecimento encerrado")
    
    def listar_rostos(self):
        if len(self.nomes_conhecidos) == 0:
            print("\nNenhum rosto cadastrado")
        else:
            print(f"\n=== Rostos Cadastrados ({len(self.nomes_conhecidos)}) ===")
            for i, nome in enumerate(self.nomes_conhecidos, 1):
                print(f"{i}. {nome}")
    
    def remover_rosto(self, nome):
        if nome in self.nomes_conhecidos:
            index = self.nomes_conhecidos.index(nome)
            self.nomes_conhecidos.pop(index)
            self.rostos_conhecidos.pop(index)
            self.salvar_encodings()
            print(f"{nome} removido com sucesso")
            return True
        else:
            print(f"{nome} nao encontrado")
            return False
    
    def testar_similaridades(self):
        if len(self.rostos_conhecidos) < 2:
            print("\nCadastre pelo menos 2 rostos para testar similaridades")
            return
        
        print("\n=== Matriz de Similaridades ===")
        print("(Valores proximos de 1.0 = muito similares)")
        print("(Valores abaixo de 0.65 = diferentes)\n")
        
        print("".ljust(15), end="")
        for nome in self.nomes_conhecidos:
            print(f"{nome[:12]:>12}", end=" ")
        print()
        
        for i, nome1 in enumerate(self.nomes_conhecidos):
            print(f"{nome1[:15]:15}", end="")
            for j, nome2 in enumerate(self.nomes_conhecidos):
                if i == j:
                    print(f"{'1.000':>12}", end=" ")
                else:
                    sim = cosine_similarity(
                        self.rostos_conhecidos[i].reshape(1, -1),
                        self.rostos_conhecidos[j].reshape(1, -1)
                    )[0][0]
                    print(f"{sim:>12.3f}", end=" ")
            print()
        
        print("\nDica: Se dois rostos diferentes tem similaridade > 0.65,")
        print("   o sistema pode confundi-los. Tente recadastrar com melhor iluminacao.")
        print("\nValores tipicos:")
        print("   - Mesma pessoa: 0.70 - 0.95")
        print("   - Pessoas diferentes: 0.30 - 0.60")


def menu_principal():
    sistema = ReconhecimentoFacial()
    
    while True:
        print("\n" + "="*50)
        print("SISTEMA DE RECONHECIMENTO FACIAL")
        print("="*50)
        print("1. Cadastrar novo rosto")
        print("2. Reconhecer rostos (tempo real)")
        print("3. Listar rostos cadastrados")
        print("4. Remover rosto")
        print("5. Testar similaridades (debug)")
        print("6. Sair")
        print("="*50)
        
        opcao = input("Escolha uma opcao: ").strip()
        
        if opcao == "1":
            nome = input("\nDigite o nome da pessoa: ").strip()
            if nome:
                sistema.cadastrar_rosto(nome)
            else:
                print("Nome invalido")
        
        elif opcao == "2":
            sistema.reconhecer_tempo_real()
        
        elif opcao == "3":
            sistema.listar_rostos()
        
        elif opcao == "4":
            sistema.listar_rostos()
            nome = input("\nDigite o nome da pessoa a remover: ").strip()
            if nome:
                sistema.remover_rosto(nome)
        
        elif opcao == "5":
            sistema.testar_similaridades()
        
        elif opcao == "6":
            print("\nAte logo!")
            break
        
        else:
            print("Opcao invalida")


if __name__ == "__main__":
    print("Inicializando sistema de reconhecimento facial...")
    menu_principal()
