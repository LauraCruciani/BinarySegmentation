import os
import cv2
import numpy as np
from tqdm import tqdm

def ricomponi(p1, p2, p3, p4, p5, p6, p7, p8, larghezza, altezza):
    nuova_immagine = np.zeros((2 * altezza, 4 * larghezza, 3), dtype=np.uint8)

    nuova_immagine[:altezza, :larghezza] = p1
    nuova_immagine[:altezza, larghezza:2 * larghezza] = p2
    nuova_immagine[:altezza, 2 * larghezza:3 * larghezza] = p3
    nuova_immagine[:altezza, 3 * larghezza:] = p4

    nuova_immagine[altezza:, :larghezza] = p8
    nuova_immagine[altezza:, larghezza:2 * larghezza] = p5
    nuova_immagine[altezza:, 2 * larghezza:3 * larghezza] = p6
    nuova_immagine[altezza:, 3 * larghezza:] = p7

    return nuova_immagine


def taglia_immagine(file_immagine):
    img = cv2.imread(file_immagine)
    altezza, larghezza, _ = img.shape
    taglio_1 = larghezza // 4
    taglio_2 = 2 * larghezza // 4
    taglio_3 = 3 * larghezza // 4

    parte_1 = img[:, :taglio_1, :]
    parte_3 = img[:, taglio_2:taglio_3, :]
    parte_4 = img[:, taglio_3:, :]

    return parte_1, parte_3, parte_4


def crea_video_da_immagini(cartella_immagini, nome_video='output_video.avi', fps=1):
    immagini = []

    for filename in os.listdir(cartella_immagini):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            percorso_immagine = os.path.join(cartella_immagini, filename)
            immagini.append(percorso_immagine)

    if not immagini:
        print(f"Images not found in '{cartella_immagini}'.")
        return

    riferimento = cv2.imread(immagini[0])
    altezza, larghezza, _ = riferimento.shape

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(nome_video, codec, fps, (larghezza, altezza))

    for percorso_immagine in immagini:
        img = cv2.imread(percorso_immagine)
        video_writer.write(img)

    video_writer.release()

    print(f"Video '{nome_video}' successfully created.")


def sovrappone_maschera(img, maschera):
    # Resize the mask to match the dimensions of the image
    #maschera_resized = cv2.resize(maschera, (img.shape[1], img.shape[0]))
    #print(img.shape, maschera.shape)
    alpha = 0.4
    masked_img = cv2.addWeighted(img, 1, maschera, alpha,0)
    return masked_img


def compute_confusion_matrix(gt, pred):
    tp = np.sum(np.logical_and(gt == 1, pred == 1))
    fp = np.sum(np.logical_and(gt == 0, pred == 1))
    fn = np.sum(np.logical_and(gt == 1, pred == 0))
    tn = np.sum(np.logical_and(gt == 0, pred == 0))
    return tp, fp, fn, tn

def put_name(img, name,font_scale=2, font_thickness=2, text_color=(0, 0, 255) ):
    text_org_acc = (10, 100)
    img = cv2.putText(img, f"{name}", text_org_acc, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                      text_color, font_thickness, cv2.LINE_AA)
    return img


def put_Text(img, acc, dsc, font_scale=2, font_thickness=2, text_color=(255, 255, 255)):
    text_org_acc = (10, img.shape[0] - 100)
    text_org_dsc = (10, img.shape[0] - 20)

    img = cv2.putText(img, f"ACC: {acc:.5f}", text_org_acc, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                      text_color, font_thickness, cv2.LINE_AA)
    img = cv2.putText(img, f"DSC: {dsc:.5f}", text_org_dsc, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                      text_color, font_thickness, cv2.LINE_AA)

    return img


def accuracy(gt, pred):
    correct_pixels = np.sum(gt == pred)
    total_pixels = gt.size
    accuracy = correct_pixels / total_pixels
    return accuracy


def dice_similarity_coefficient(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dsc = (2. * intersection) / (union + 1e-6)
    return dsc


def save_results(name, dsc, accuracy, f1score):




if __name__ == "__main__":
    # QUESTO VA CONTROLLATO OGNI VOLTA
    name=["UNET", "UNET++", "LadderNet", "AttentionUnet", "SETR"]
    cartella1 = "risultati/Unet_final/result_img_Unet"
    cartella2="risultati/Unet++/result_img_Unet++"
    cartella3="risultati/LadderNet/result_img_LadderNet"
    cartella4="risultati/Attention_unet/result_img_Attention_Unet"
    cartella5="risultati/SETR/result_img_SETR"
    cartelle=[cartella1,cartella2,cartella3,cartella4,cartella5]

    for k in range(len(cartelle)):
        formattato=cartelle[k]
        if (os.path.exists(formattato) is False):
            print(f"{formattato} doesn't exist")
            break
        else: 
            if not os.path.exists("img_final"):
                os.makedirs("img_final")
    paths=[]
    for filename in tqdm(os.listdir(cartella1)):
        paths=[]
        dsc=[]
        acc=[]
        if filename.endswith(".jpg") or filename.endswith(".png"):
            for k in range(len(cartelle)):
                #print(cartelle[k])
                paths.append(os.path.join(cartelle[k], filename))
                preds=[]
                dsc=[]
                acc=[]
                for j in range (len(paths)):
                    if j==0:
                        img, _, gt=taglia_immagine(paths[j])
                    _, pred,_=taglia_immagine(paths[j])
                    predd=pred/255
                    gtt=gt/255
                    accu=accuracy(gtt, predd)
                    dscoeff=dice_similarity_coefficient(gtt,predd)

                    preds.append(pred)
                    acc.append(accu)
                    dsc.append(dscoeff)
            pos_max= np.argmax(dsc)
            #print(filename, len(preds), pos_max, len(acc))

            gt2=gt.copy()
            img6=sovrappone_maschera(img,gt2)
            for i in range(len(preds)):
                preds[i]=put_name(preds[i], name[i])
                if i==pos_max:
                    preds[i] = put_Text(preds[i], acc[i], dsc[i], text_color=(0,255,0)) 
                else: preds[i] = put_Text(preds[i], acc[i], dsc[i]) 
            gt=put_name(gt,"GT")

            ## VA MODIFICATO QUI            
            img_finale=ricomponi(img,preds[0],preds[1],preds[2],preds[3], preds[4], gt, img6, 1280, 960)
            ########################
            cv2.imwrite("img_final/" + os.path.basename(filename), img_finale)

        
    cartella_immagini = "img_final"
    crea_video_da_immagini(cartella_immagini)
                