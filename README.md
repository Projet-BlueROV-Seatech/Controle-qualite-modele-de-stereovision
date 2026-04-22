# 📷 Outil de Contrôle Qualité Stéréo 3D
Simulateur interactif développé en Python par une IA pour évaluer la qualité de la vision par IA. 
L'outil simule mathématiquement un système de caméras et utilise la projection sténopé ainsi que la triangulation DLT (Direct Linear Transform) par SVD.

---

## ✨ Fonctionnalités
Vue 3D interactive : Représentation spatiale du pavé droit, des centres optiques et des cônes de vision des deux caméras.

Projections 2D : Simulation réaliste de la vue des caméras 1 et 2 en temps réel.

Contrôle paramétrique : Sliders interactifs pour ajuster la position spatiale (X, Y, Z) et l'angle de lacet (yaw) de l'objet inspecté.

Triangulation manuelle : Interface "cliquer-glisser" permettant de dessiner des boîtes englobantes (bounding boxes) sur les flux caméras pour cibler le centre de l'objet.

Calcul d'erreurs : Retour immédiat sur l'écart entre la position réelle et la position estimée (erreur euclidienne 3D, erreur angulaire et décalage en pixels).

---

## 🛠️ Prérequis
Le script repose uniquement sur les bibliothèques scientifiques standards de Python. Pour installer les dépendances : "pip install numpy matplotlib"

---

## 🚀 Utilisation
Lancez simplement le script via votre terminal ou autre : "python qualite_stereo_3d.py"

Mode d'emploi de l'interface :

🎛️ Ajustez la position et l'angle du pavé à l'aide des curseurs situés en bas à gauche.

🖱️ Encadrez le pavé en effectuant un cliquer-glisser (dessin d'un rectangle) sur chacune des deux vues caméra à droite.

🎯 Cliquez sur le bouton "▶ Trianguler".

📊 Analysez les résultats : la position estimée apparaît en vert dans la vue 3D et le détail des erreurs s'affiche dans le panneau inférieur.
