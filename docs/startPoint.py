import speech_recognition as sr
import time
import pandas as pd
from vocab import words
from timed import df_emo, df_62, df_71
import numpy as np
from cv2 import cv2
import datetime as dt
import math
import simpleaudio as sa
import itertools
import collections


r = sr.Recognizer()
def capture():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #print('Ready')
        audio = r.listen(source)
    return audio
    

def process_text(capt_text):
    try: 
        text = r.recognize_google(capt_text)
        words = text.split()
        return words
    except sr.UnknownValueError:
        print('Google audio failed')
    except sr.RequestError as e:
        print('Google Req failed')
    


t = dt.datetime.now()

face_cascade_path = r'haarcascade_frontalface_default.xml'
eye_cascade_path = r'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

video_capture = cv2.VideoCapture(0)

### Timed Program Integration ###
X1, y1 = df_emo.iloc[:, :].values, df_62.iloc[:35, :].values 
X2, y2 = df_62.iloc[:, :].values, df_71.iloc[:471, :].values 

O0 = [] #X1
O0.append(X1[:, 0])
O0.append(X1[:, 1])
O0.append(X1[:, 2])
O0.append(X1[:, 3])
Oe = [] #Odd Emotion
for num in O0:
    for i in num:
        Oe.append(i)

E0 = [] #X2
E0.append(X2[:, 0])
E0.append(X2[:, 1])
E0.append(X2[:, 2])
E0.append(X2[:, 3])
E7 = [] #Even Objects
for num in E0:
    for i in num:
        E7.append(i)

# Y1
E0 = []
E0.append(y1[:, 0])
Em6 = [] #Even motion 6
for num in E0:
    for i in num:
        Em6.append(i)

O0 = []
O0.append(y1[:, 1])
Om6 = [] #Odd motion 6
for num in O0:
    for i in num:
        Om6.append(i)

E0 = []
E0.append(y1[:, 2])
Em2 = [] #Even motion 2
for num in E0:
    for i in num:
        Em2.append(i)

O0 = []
O0.append(y1[:, 3])
Om2 = [] #Odd motion 2
for num in O0:
    for i in num:
        Om2.append(i)

#Y2
E0 = []
E0.append(y2[:, 0])
Em67 = [] #Even motion 6-7
for num in E0:
    for i in num:
        Em67.append(i)

O0 = []
O0.append(y2[:, 1])
Om61 = [] #Odd motion 6-1
for num in O0:
    for i in num:
        Om61.append(i)

E0 = []
E0.append(y2[:, 2])
Em27 = [] #Even motion 2-7
for num in E0:
    for i in num:
        Em27.append(i)

O0 = []
O0.append(y2[:, 3])
Om21 = [] #Odd motion 2-1
for num in O0:
    for i in num:
        Om21.append(i)

#Haves
        #Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21 Em27
def function1(X1,X2,Y1,Y2):
    OIO = []
    for num in itertools.islice(X2, 0, 140):
            OIO.append(num)
    A10 = [OIO_i / X1_i for OIO_i, X1_i in zip(OIO, X1)]#Oe+OIO
    B10 = [Y1_i / Y2_i for Y1_i, Y2_i in zip(Y1, Y2)]#E7+Td
    return A10,B10
def function2(X1,X2,Y1,Y2):
    Td = []
    for num in itertools.islice(X2, 0, 35):
        Td.append(num)
    C11 = [X1_i + Td_i for X1_i, Td_i in zip(X1, Td)] #Em6+Td
    OIO = []
    for num in itertools.islice(Y2, 0, 35):
        OIO.append(num)
    D13 = [Y1_i + OIO_i for Y1_i, OIO_i in zip(Y1, OIO)] #Om6+OIO
    return C11,D13
def function3(X1,X2,Y1,Y2):
    OIO = []
    for num in itertools.islice(X2, 0, 35):
        OIO.append(num)
    E14 = [X1_i + OIO_i for X1_i, OIO_i in zip(X1, OIO)] #Om2+OIO
    Td = []
    for num in itertools.islice(Y2, 0, 35):
        Td.append(num)
    F14 = [Y1_i + Td_i for Y1_i, Td_i in zip(Y1, Td)] #Em2+Td
    return E14,F14
def function4(X1,X2,Y1):
    Td = []
    for num in itertools.islice(X2, 0, 471):
        Td.append(num)
    G11 = [X1_i - Td_i for X1_i, Td_i in zip(X1, Td)]#X1 - X2 #Em67-Td
    H12 = [Y1_i - Td_i for Y1_i, Td_i in zip(Y1, Td)]#Y1 - Y2 #Om61-Td
    return G11,H12
def function5(X1,X2,Y1,Y2):
    OIO = []
    for num in itertools.islice(X2, 0, 471):
        OIO.append(num)
    Td = []
    for num in itertools.islice(Y2, 0, 471):
        Td.append(num)
    I12 = [X1_i - OIO_i for X1_i, OIO_i in zip(X1, OIO)]#X1 - X2 #Om21-OIO
    J14 = [Y1_i - Td_i for Y1_i, Td_i in zip(Y1, Td)]#Y1 - Y2 #Em27-Td
    return I12,J14
'''
function1(Oe,OIO,E7,Td)
function2(Em6, Td, Om6, OIO)
function3(Om2, OIO, Em2, Td)
function4(Em67, Td, Om61, Td)
function5(Om21, OIO, Em27, Td)
'''
def find_god(X1, X2, X5):
    DO13 = [X1_i + X2_i for X1_i, X2_i in zip(X1, X2)]
    DOC24 = [X2_i / DO13_i for X2_i, DO13_i in zip(X2, DO13)]
    GOD24 = [X5_i / DO13_i for X5_i, DO13_i in zip(X5, DO13)]
    return DO13, DOC24, GOD24

def God(X1,X2,X3,X4):
    DO = []
    for num in itertools.islice(X1, 0, 35):
        DO.append(num)
    GO = []
    for num in itertools.islice(X2, 0, 35):
        GO.append(num)
    DONT = [DO_i + GO_i for DO_i, GO_i in zip(DO, GO)]
    GOD = [X3_i / X4_i for X3_i, X4_i in zip(X3, X4)]
    return DONT, GOD


#Word PreProcessing thought endeavors
def BrainFunction(Coded, Person, Hope, More, Business1, WHA, WSA, Coded1):
    POLD = []
    HOLD = []
    MOLD = []
    for let in Coded:
        Pnum = [Person_i / let_i for Person_i, let_i in zip(Person, let)]
        POLD.append(Pnum)
        Hnum = [Hope_i + let_i for Hope_i, let_i in zip(Hope, let)]
        HOLD.append(Hnum)
        Mnum = [More_i - let_i for More_i, let_i in zip(More, let)]
        MOLD.append(Mnum)
    #Idea Formation
    Idea = []
    for POLD_i, HOLD_i, MOLD_i in zip(POLD, HOLD, MOLD):
        ID = [POLD_j / HOLD_j * MOLD_j for POLD_j, HOLD_j, MOLD_j in zip(POLD_i, HOLD_i, MOLD_i)]
        Idea.append(ID)
    #Felt Progessives
    Fidea = []
    for let in Idea:
        Bus = [Business1_i + let_i for Business1_i, let_i in zip(Business1, let)]
        Fidea.append(Bus)
    letter0 = []
    for let in Fidea:
        num0 = [Hope_i / (let_i * (WSA_i + WHA_i)) / More_i for Hope_i, WSA_i, let_i, WHA_i, More_i in zip(Hope, WSA, let, WHA, More)]
        letter0.append(num0)
    #Complex 3D List for word comprehensives
    spot = 0
    spread = []
    prim = []
    #print(len(Coded1))
    for num in Coded1:
        aug = -1
        for let in itertools.islice(letter0, spot, spot+num):
            aug += 1
            if aug == 0:
                spread.append(let)
            elif aug == 1:
                spread.append(let)
            elif aug == 2:
                spread.append(let)
            elif aug == 3:
                spread.append(let)
            elif aug == 4:
                spread.append(let)
            elif aug == 5:
                spread.append(let)
            elif aug == 6:
                spread.append(let)
            elif aug == 7:
                spread.append(let)
            elif aug == 8:
                spread.append(let)
            elif aug == 9:
                spread.append(let)
            elif aug == 10:
                spread.append(let)
            elif aug == 11:
                spread.append(let)
            elif aug == 12:
                spread.append(let)
            elif aug == 13:
                spread.append(let)
            elif aug == 14:
                spread.append(let)
            
            
        prim.append(spread)
        spread = []
        spot += num

    #print(prim[0:3])
    return prim

#Emotional Start
def Heart(Hoe1, Mo, Re, Person1, Business):
    #Emotion Start
    EF = [Re_i / Person1_i * Hoe1_i for Person1_i, Hoe1_i, Re_i in zip(Person1, Hoe1, Re)]
    Care = [EF_i / Hoe1_i / Mo_i * Mo_i for EF_i, Hoe1_i, Mo_i in zip(EF, Hoe1, Mo)]
    Brave = [(Mo_i * Hoe1_i - Re_i / EF_i) * Hoe1_i for Mo_i, Hoe1_i, Re_i, EF_i in zip(Mo, Hoe1, Re, EF)]
    Hate = [(EF_i * Mo_i * Hoe1_i / Re_i) / Mo_i for EF_i, Mo_i, Hoe1_i, Re_i in zip(EF, Mo, Hoe1, Re)]
    Fear = [(Hoe1_i * Mo_i - EF_i / Re_i) / Hoe1_i for Hoe1_i, Mo_i, EF_i, Re_i in zip(Hoe1, Mo, EF, Re)]
    NEC = [Re_i * Re_i * (Hoe1_i * (Mo_i / Re_i) / EF_i) * Hoe1_i for Mo_i, Re_i, EF_i, Hoe1_i in zip(Mo, Re, EF, Hoe1)]

    Business1 = [Business_i / ((Care_i / Hate_i / Fear_i - Brave_i) * NEC_i) for Business_i, Hate_i, Care_i, Fear_i, Brave_i, NEC_i in zip(Business, Hate, Care, Fear, Brave, NEC)]
    
    return Business1, Care, Brave, Hate, Fear, NEC

#Fluidity Start
def Genitalia(Om21, Em67, Society, OIO, Td, Om61, Em27):
    Gen = [Td_i / OIO_i for Td_i, OIO_i in zip(Td, OIO)]
    #print(Gen)
    Left = [Om21_i * OIO_i for Om21_i, OIO_i in zip(Om21, OIO)]
    Right = [Em67_i + Td_i for Em67_i, Td_i in zip(Em67, Td)]
    Fluid = [Om21_i - Em67_i * OIO_i for Om21_i, Em67_i, OIO_i in zip(Om21, Em67, OIO)]
    Society1 = [(Left_i + Right_i - Gen_i * Fluid_i) / Society_i for Left_i, Right_i, Gen_i, Fluid_i, Society_i in zip(Left, Right, Gen, Fluid, Society)]
    #7-1 Spherical Air Transversal Formation
    Last = [Em67_i * Society1_i / Em27_i for Em67_i, Society1_i, Em27_i in zip(Em67, Society1, Em27)]
    First = [Om61_i * Society1_i / Om21_i for Om61_i, Society1_i, Om21_i in zip(Om61, Society1, Om21)]
    return Last, First

#Preliminary Word Encoding
def WordCode(words, Em6, Om6, Em2, Om2):
    OLD0 = []
    OLD1 = []
    for word in words:
        L = 0
        for letter in word:
            L += 1
            if L == len(word):
                OLD1.append(L)
    for word in words:
        for letter in word:
            if letter == 'a':
                A = [Om2_i * Em6_i for Om2_i, Em6_i in zip(Om2, Em6)]
                OLD0.append(A)
            elif letter == 'b':
                B = [Om6_i - Em2_i / Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(B)
            elif letter == 'c':
                C = [Om6_i + Em2_i / Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(C)
            elif letter == 'd':
                D = [Om6_i - Em2_i * Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(D)
            elif letter == 'e':
                E = [Em6_i / Om6_i for Em6_i, Om6_i in zip(Em6, Om6)]
                OLD0.append(E)
            elif letter == 'f':
                F = [Om6_i + Em2_i * Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(F)
            elif letter == 'g':
                G = [Om2_i * Em6_i + Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(G)
            elif letter == 'h':
                H = [Om2_i * Em6_i - Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(H)
            elif letter == 'i':
                I = [Om6_i + Em2_i for Om6_i, Em2_i in zip(Om6, Em2)]
                OLD0.append(I)
            elif letter == 'j':
                J = [Om2_i / Em6_i + Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(J)
            elif letter == 'k':
                K = [Om2_i / Em6_i - Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(K)
            elif letter == 'l':
                L = [Om6_i / Em2_i + Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(L)
            elif letter == 'm':
                M = [Om6_i / Em2_i - Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(M)
            elif letter == 'n':
                N = [Om6_i * Em2_i + Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(N)
            elif letter == 'o':
                O = [Om6_i - Em6_i for Om6_i, Em6_i in zip(Om6, Em6)]
                OLD0.append(O)
            elif letter == 'p':
                P = [Om6_i * Em2_i - Om2_i for Om6_i, Em2_i, Om2_i in zip(Om6, Em2, Om2)]
                OLD0.append(P)
            elif letter == 'q':
                Q = [Om6_i * Em6_i + Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(Q)
            elif letter == 'r':
                R = [Om6_i * Em6_i - Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(R)
            elif letter == 's':
                S = [Om6_i / Em6_i + Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(S)
            elif letter == 't':
                T = [Om6_i / Em6_i - Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(T)
            elif letter == 'u':
                U = [Em2_i + Om2_i for Em2_i, Om2_i in zip(Em2, Om2)]
                OLD0.append(U)
            elif letter == 'v':
                V = [Om2_i + Em2_i / Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(V)
            elif letter == 'w':
                W = [Om2_i - Em2_i / Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(W)
            elif letter == 'x':
                X = [Om2_i + Em2_i * Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(X)
            elif letter == 'y':
                Y = [Om2_i + Em2_i * Em6_i - Om6_i for Om2_i, Em2_i, Em6_i, Om6_i in zip(Om2, Em2, Em6, Om6)]
                OLD0.append(Y)
            elif letter == 'z':
                Z = [Om2_i - Em2_i * Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(Z)
            elif letter == '0':
                Zero = [Om2_i + Em6_i * Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(Zero)
            elif letter == '1':
                One = [Om2_i - Em6_i * Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(One)
            elif letter == '2':
                Two = [Om2_i + Em6_i / Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(Two)
            elif letter == '3':
                Three = [Om2_i - Em6_i / Om6_i for Om2_i, Em6_i, Om6_i in zip(Om2, Em6, Om6)]
                OLD0.append(Three)
            elif letter == '4':
                Four = [Om6_i + Em6_i * Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(Four)
            elif letter == '5':
                Five = [Om6_i - Em6_i * Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(Five)
            elif letter == '6':
                Six = [Om6_i + Em6_i / Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(Six)
            elif letter == '7':
                Seven = [Om6_i - Em6_i / Om2_i for Om6_i, Em6_i, Om2_i in zip(Om6, Em6, Om2)]
                OLD0.append(Seven)
            elif letter == '8':
                Eight = [Om2_i / Em2_i + Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(Eight)
            elif letter == '9':
                Nine = [Om2_i / Em2_i - Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(Nine)
            elif letter == '\'':
                Ni = [Om2_i * Em2_i + Om6_i for Om2_i, Em2_i, Om6_i in zip(Om2, Em2, Om6)]
                OLD0.append(Ni)
    return OLD0, OLD1
OLD0, OLD1 = WordCode(words, Em6, Om6, Em2, Om2)
#print(OLD0[0:10])

#Use Genitalia and Heart Outputs
def EmoteEntangle(Last, First, Hate, Care, Fear, Brave, NEC, I12, G11, H12, J14, Em6, Om6, Om2, Em2):
    Two = [I12_i - J14_i for I12_i, J14_i in zip(I12, J14)]
    Six = [G11_i / H12_i for G11_i, H12_i in zip(G11, H12)]

    Hate1 = [Last_i * Six_i - Hate_i for Last_i, Six_i, Hate_i in zip(Last, Six, Hate)]
    Hate2 = [First_i * Six_i - Hate_i for First_i, Six_i, Hate_i in zip(First, Six, Hate)]
    Hate3 = [Last_i * Six_i + Hate_i for Last_i, Six_i, Hate_i in zip(Last, Six, Hate)]
    Hate4 = [First_i * Six_i + Hate_i for First_i, Six_i, Hate_i in zip(First, Six, Hate)]

    Care1 = [Last_i / Two_i - Care_i for Last_i, Two_i, Care_i in zip(Last, Two, Care)]
    Care2 = [First_i / Two_i - Care_i for First_i, Two_i, Care_i in zip(First, Two, Care)]
    Care3 = [Last_i / Two_i + Care_i for Last_i, Two_i, Care_i in zip(Last, Two, Care)]
    Care4 = [First_i / Two_i + Care_i for First_i, Two_i, Care_i in zip(First, Two, Care)]

    Fear1 = [Last_i * Six_i - Fear_i for Last_i, Six_i, Fear_i in zip(Last, Six, Fear)]
    Fear2 = [First_i * Six_i - Fear_i for First_i, Six_i, Fear_i in zip(First, Six, Fear)]
    Fear3 = [Last_i * Six_i + Fear_i for Last_i, Six_i, Fear_i in zip(Last, Six, Fear)]
    Fear4 = [First_i * Six_i + Fear_i for First_i, Six_i, Fear_i in zip(First, Six, Fear)]

    Brave1 = [Last_i / Two_i - Brave_i for Last_i, Two_i, Brave_i in zip(Last, Two, Brave)]
    Brave2 = [First_i / Two_i - Brave_i for First_i, Two_i, Brave_i in zip(First, Two, Brave)]
    Brave3 = [Last_i / Two_i + Brave_i for Last_i, Two_i, Brave_i in zip(Last, Two, Brave)]
    Brave4 = [First_i / Two_i + Brave_i for First_i, Two_i, Brave_i in zip(First, Two, Brave)]

    NEC1 = [Last_i * Six_i - NEC_i for Last_i, Six_i, NEC_i in zip(Last, Six, NEC)]
    NEC2 = [First_i * Six_i - NEC_i for First_i, Six_i, NEC_i in zip(First, Six, NEC)]
    NEC3 = [Last_i + Six_i * Two_i - NEC_i for Last_i, Six_i, Two_i, NEC_i in zip(Last, Six, Two, NEC)]
    NEC4 = [First_i + Six_i * Two_i - NEC_i for First_i, Six_i, Two_i, NEC_i in zip(First, Six, Two, NEC)]
    NEC5 = [Last_i + Six_i * Two_i + NEC_i for Last_i, Six_i, Two_i, NEC_i in zip(Last, Six, Two, NEC)]
    NEC6 = [First_i + Six_i * Two_i + NEC_i for First_i, Six_i, Two_i, NEC_i in zip(First, Six, Two, NEC)]
    NEC7 = [Last_i * Six_i + NEC_i for Last_i, Six_i, NEC_i in zip(Last, Six, NEC)]
    NEC8 = [First_i * Six_i + NEC_i for First_i, Six_i, NEC_i in zip(First, Six, NEC)]
    NEC9 = [Last_i / Two_i - NEC_i for Last_i, Two_i, NEC_i in zip(Last, Two, NEC)]
    NEC10 = [First_i / Two_i - NEC_i for First_i, Two_i, NEC_i in zip(First, Two, NEC)]
    NEC11 = [Last_i / Two_i + NEC_i for Last_i, Two_i, NEC_i in zip(Last, Two, NEC)]
    NEC12 = [First_i / Two_i + NEC_i for First_i, Two_i, NEC_i in zip(First, Two, NEC)]

    HEF1 = [Last_i * First_i * Six_i - Hate_i for Last_i, First_i, Six_i, Hate_i in zip(Last, First, Six, Hate)]
    HEF2 = [Last_i * First_i * Six_i + Hate_i for Last_i, First_i, Six_i, Hate_i in zip(Last, First, Six, Hate)]

    CEF1 = [Last_i * First_i / Two_i - Care_i for Last_i, First_i, Two_i, Care_i in zip(Last, First, Two, Care)]
    CEF2 = [Last_i * First_i / Two_i + Care_i for Last_i, First_i, Two_i, Care_i in zip(Last, First, Two, Care)]

    FEF1 = [Last_i * First_i * Six_i / Fear_i for Last_i, First_i, Six_i, Fear_i in zip(Last, First, Six, Fear)]
    FEF2 = [Last_i * First_i * Six_i * Fear_i for Last_i, First_i, Six_i, Fear_i in zip(Last, First, Six, Fear)]

    BEF1 = [Last_i * First_i / Two_i / Brave_i for Last_i, First_i, Two_i, Brave_i in zip(Last, First, Two, Brave)]
    BEF2 = [Last_i * First_i / Two_i * Brave_i for Last_i, First_i, Two_i, Brave_i in zip(Last, First, Two, Brave)]

    SaF = [(Fear3_i + Fear4_i) / (HEF1_i - FEF2_i) for Fear3_i, Fear4_i, HEF1_i, FEF2_i in zip(Fear3, Fear4, HEF1, FEF2)]
    HaF = [(Brave3_i + Brave4_i) / (CEF1_i - BEF2_i) for Brave3_i, Brave4_i, CEF1_i, BEF2_i in zip(Brave3, Brave4, CEF1, BEF2)]

    Hate = [Hate1_i * Hate2_i / (HEF1_i - HEF2_i) / Hate3_i + Hate4_i for Hate1_i, Hate2_i, HEF1_i, HEF2_i, Hate3_i, Hate4_i in zip(Hate1, Hate2, HEF1, HEF2, Hate3, Hate4)]
    Care = [Care1_i * Care2_i / (CEF1_i - CEF2_i) / Care3_i + Care4_i for Care1_i, Care2_i, CEF1_i, CEF2_i, Care3_i, Care4_i in zip(Care1, Care2, CEF1, CEF2, Care3, Care4)]
    Fear = [Fear1_i * Fear2_i / (FEF1_i - FEF2_i) / Fear3_i + Fear4_i for Fear1_i, Fear2_i, FEF1_i, FEF2_i, Fear3_i, Fear4_i in zip(Fear1, Fear2, FEF1, FEF2, Fear3, Fear4)]
    Brave = [Brave1_i * Brave2_i / (BEF1_i - BEF2_i) / Brave3_i + Brave4_i for Brave1_i, Brave2_i, BEF1_i, BEF2_i, Brave3_i, Brave4_i in zip(Brave1, Brave2, BEF1, BEF2, Brave3, Brave4)]
    SNEC = [NEC1_i * NEC2_i / (NEC3_i - NEC4_i) / NEC7_i + NEC8_i for NEC1_i, NEC2_i, NEC3_i, NEC4_i, NEC7_i, NEC8_i in zip(NEC1, NEC2, NEC3, NEC4, NEC7, NEC8)]
    HNEC = [NEC11_i * NEC12_i / (NEC5_i - NEC6_i) / NEC9_i + NEC10_i for NEC11_i, NEC12_i, NEC5_i, NEC6_i, NEC9_i, NEC10_i in zip(NEC11, NEC12, NEC5, NEC6, NEC9, NEC10)]


    WH = Two = [Em6_i / Hate_i for Em6_i, Hate_i in zip(Em6, Hate)]
    WC = [Em2_i / Care_i for Em2_i, Care_i in zip(Em2, Care)]
    WF = [Om6_i / Fear_i for Om6_i, Fear_i in zip(Om6, Fear)]
    WB = [Om2_i / Brave_i for Om2_i, Brave_i in zip(Om2, Brave)]

    WSNEC = [(Om6_i - Om2_i) / SNEC_i for Om6_i, Om2_i, SNEC_i in zip(Om6, Om2, SNEC)]
    WHNEC = [(Om6_i - Om2_i) / HNEC_i for Om6_i, Om2_i, HNEC_i in zip(Om6, Om2, HNEC)]

    WSA = [(Em6_i * Om6_i) / SaF_i for Em6_i, Om6_i, SaF_i in zip(Em6, Om6, SaF)]
    WHA = [(Em2_i * Om2_i) / HaF_i for Em2_i, Om2_i, HaF_i in zip(Em2, Om2, HaF)]
    return WH, WC, WF, WB, WSNEC, WHNEC, WSA, WHA




def DataInt(X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, OIO, Td, contours, hierarchy, W0=[], W1=[]):
    A10, B10 = function1(X3,OIO,X4,Td)
    #print(A10)
    #print(B10)
    C11, D13 = function2(X5, Td, X6, OIO)
    E14, F14 = function3(X7, OIO, X8, Td)
    G11, H12 = function4(X9, Td, X10)
    I12, J14 = function5(X11, OIO, X12, Td)
    DO13, DOC24, GOD24 = find_god(H12, I12, G11)
    
    
    
    DOCF38 = [DOC24_i + G11_i for DOC24_i, G11_i in zip(DOC24, G11)]#DOC24, G11
    GODE38 = [GOD24_i + I12_i for GOD24_i, I12_i in zip(GOD24, I12)] #GOD24, E14
    Door = [DOCF38_i / GODE38_i for DOCF38_i, GODE38_i in zip(DOCF38, GODE38)]
    #print(Door)
#Preliminary Idea Generation
    DONT, GOD = God(DOC24, GOD24, E14, F14)
    Society = [god / 2 for god in GODE38]
    Business = [soc * soc * soc * soc for soc in E14]
    SOC = []
    for num in itertools.islice(Society, 0, 35):
        SOC.append(num)
#FEELING Separations (Empty Feeling, Feeling)
    Person1 = [GOD_i - SOC_i for GOD_i, SOC_i in zip(GOD, SOC)]
    Person = [SOC_i - Business_i for SOC_i, Business_i in zip(SOC, Business)]
    #Building Grounds
    OIO1 = []
    for num in itertools.islice(OIO, 0, 471):
        OIO1.append(num)
    Hoe = [H12_i + OIO1_i for H12_i, OIO1_i in zip(H12, OIO1)]
    Hoe1 = []
    for num in itertools.islice(Hoe, 0, 35):
        Hoe1.append(num)
#BELIEF
    Hope = [E14_i + Hoe1_i for E14_i, Hoe1_i in zip(E14, Hoe1)]
    OIO2 = []
    for num in itertools.islice(OIO, 0, 35):
        OIO2.append(num)
    Mo = [E14_i - OIO2_i for E14_i, OIO2_i in zip(E14, OIO2)]
    Re = [Business_i - Td_i for Business_i, Td_i in zip(Business, Td)]
#CREATION
    More = [Mo_i / Re_i for Mo_i, Re_i in zip(Mo, Re)]
    #7-1 Transversal
    Last, First = Genitalia(X11, X9, Society, OIO1, Td, X10, X12)
    Business1, Care, Brave, Hate, Fear, NEC = Heart(Hoe1, Mo, Re, Person1, Business)
    #Number Area Systematic Continuations
    WH, WC, WF, WB, WSNEC, WHNEC, WSA, WHA = EmoteEntangle(Last, First, Hate, Care, Fear, Brave, NEC, I12, G11, H12, J14, X5, X6, X7, X8)
    
    if len(W1) > 0:
        prim = BrainFunction(W0, Person, Hope, More, Business1, WHA, WSA, W1)
    
    #Sight's Data Integrations
    Even, Odd = ContourStract(contours)
    Uno, Dos = HierarchyStract(hierarchy)
    #Separational sight integrations for Even,Odd
    NEven = []
    for lis1 in Even:
        new1 = [lis1_i / B10_i for lis1_i, B10_i in zip(lis1, B10)]
        NEven.append(new1)
    #print(len(NEven))
    NOdd = []
    for lis2 in Odd:
        new2 = [lis2_i / B10_i for lis2_i, B10_i in zip(lis2, B10)]
        NOdd.append(new2)
    #print(len(NOdd))
    #Separational sight integrations for Uno, Dos
    NUno = []
    NDos = []
    if len(Uno)==1884 and len(Dos)==1884:
        newU = [Uno_i / B10_i for Uno_i, B10_i in zip(Uno, B10)]
        newD = [Dos_i / B10_i for Dos_i, B10_i in zip(Dos, B10)]
        NUno.append(newU)
        NDos.append(newD)
    elif len(Uno)<32 and len(Dos)<32:
        for lisU in Uno:
            NewU = [lisU_i / B10_i for lisU_i, B10_i in zip(lisU, B10)]
            NUno.append(NewU)
        for lisD in Dos:
            NewD = [lisD_i / B10_i for lisD_i, B10_i in zip(lisD, B10)]
            NDos.append(NewD)
    #Haves -- NEven, NOdd, NUno, NDos -- Integrated with B10 Dataset
    #print(len(NUno))
    #One Portionality Creation
    Sight3d = []
    Sight3d.append(NEven)
    Sight3d.append(NOdd)
    Sight3d.append(NUno)
    Sight3d.append(NDos)
    seeQ = []
    for val in Sight3d:
        for top in val:
            seeQ.append(sum(top))
    #print(len(seeQ))
    #print(seeQ)
    
    
    return A10, WH, WC, WF, WB, WSNEC, WHNEC, prim, seeQ

def FaceStract(faces1):
    FaceV1 = []
    faces1 = faces1.tolist()
    #print(faces1)
    
    for cell in faces1:
        for num in cell:
            FaceV1.append(num)
    return FaceV1

##################################CONTOURS#PROCESS#####################################################################################################################

def ContourStract(contours):
    #Whatever lists
    ET1 = []
    OT16 = []
    ET8 = []
    EL1 = []
    OL32 = []
    EL16 = []
    #Original Data Construction
    cont = []
    lines = []
    thing = 0
    odd = []
    even = []
    for num in contours:
        for thing in num:
            thing = thing.tolist()
            cont.append(thing)
    for num in cont:
        for item in num:
            for act in item:
                lines.append(act)
    for num in lines:
        if num % 2 == 0:
            even.append(num)
        elif num % 2 != 0:
            odd.append(num)
    #Splitting Portionalities
    oddL1 = []
    evenL1 = []
    oddL2 = []
    evenL2 = []
    ind = -1
    #odd splits with number 1
    for num in odd:
        ind+=1
        if ind % 2 == 0:
            evenL1.append(num)
        elif ind % 2 != 0:
            oddL1.append(num)
    ind = -1
    #Even splits with number 2
    for num in even:
        ind+=1
        if ind % 2 == 0:
            evenL2.append(num)
        elif ind % 2 != 0:
            oddL2.append(num)
    #Now have oddL1, evenL1, oddL2, evenL2 and need split again
    oddN1 = []
    evenN1 = []
    oddN2 = []
    evenN2 = []
    oddN3 = []
    evenN3 = []
    oddN4 = []
    evenN4 = []
    ind = -1
    #oddL splits with 1
    for num in oddL1:
        ind+=1
        if ind % 2 == 0:
            evenN1.append(num)
        elif ind % 2 != 0:
            oddN1.append(num)
    ind = -1
    #evenL splits with 1
    for num in evenL1:
        ind+=1
        if ind % 2 == 0:
            evenN2.append(num)
        elif ind % 2 != 0:
            oddN2.append(num)
    ind = -1
    #oddL splits with 2
    for num in oddL2:
        ind+=1
        if ind % 2 == 0:
            evenN3.append(num)
        elif ind % 2 != 0:
            oddN3.append(num)
    ind = -1
    #evenL splits with 2
    for num in evenL2:
        ind+=1
        if ind % 2 == 0:
            evenN4.append(num)
        elif ind % 2 != 0:
            oddN4.append(num)
    #Now have oddN1, evenN1, oddN2, evenN2, oddN3, evenN3, oddN4, evenN4 SPLIT AGAIN
    OV1 = []
    EV1 = []
    OV2 = []
    EV2 = []
    OV3 = []
    EV3 = []
    OV4 = []
    EV4 = []
    OV5 = []
    EV5 = []
    OV6 = []
    EV6 = []
    OV7 = []
    EV7 = []
    OV8 = []
    EV8 = []
    #oddN split with 1
    ind=-1
    for num in oddN1:
        ind+=1
        if ind % 2 == 0:
            EV1.append(num)
        elif ind % 2 != 0:
            OV1.append(num)
    #evenN split with 1
    ind=-1
    for num in evenN1:
        ind+=1
        if ind % 2 == 0:
            EV2.append(num)
        elif ind % 2 != 0:
            OV2.append(num)
    #oddN split with 2
    ind=-1
    for num in oddN2:
        ind+=1
        if ind % 2 == 0:
            EV3.append(num)
        elif ind % 2 != 0:
            OV3.append(num)
    #evenN split with 2
    ind=-1
    for num in evenN2:
        ind+=1
        if ind % 2 == 0:
            EV4.append(num)
        elif ind % 2 != 0:
            OV4.append(num)
    #oddN split with 3
    ind=-1
    for num in oddN3:
        ind+=1
        if ind % 2 == 0:
            EV5.append(num)
        elif ind % 2 != 0:
            OV5.append(num)
    #evenN split with 3
    ind=-1
    for num in evenN3:
        ind+=1
        if ind % 2 == 0:
            EV6.append(num)
        elif ind % 2 != 0:
            OV6.append(num)
    #oddN split with 4
    ind=-1
    for num in oddN4:
        ind+=1
        if ind % 2 == 0:
            EV7.append(num)
        elif ind % 2 != 0:
            OV7.append(num)
    #evenN split with 4
    ind=-1
    for num in evenN4:
        ind+=1
        if ind % 2 == 0:
            EV8.append(num)
        elif ind % 2 != 0:
            OV8.append(num)
    #Now have EV1, OV1, EV2, OV2, EV3, OV3, EV4, OV4, EV5, OV5, EV6, OV6, EV7, OV7, EV8, OV8
    #Needs Creation of control flow... This works if not looking outside at textures
    
    
    if len(EV1) > 1884 and len(OV8) > 1884:
        ET1 = []
        OT1 = []
        ET2 = []
        OT2 = []
        ET3 = []
        OT3 = []
        ET4 = []
        OT4 = []
        ET5 = []
        OT5 = []
        ET6 = []
        OT6 = []
        ET7 = []
        OT7 = []
        ET8 = []
        OT8 = []
        ET9 = []
        OT9 = []
        ET10 = []
        OT10 = []
        ET11 = []
        OT11 = []
        ET12 = []
        OT12 = []
        ET13 = []
        OT13 = []
        ET14 = []
        OT14 = []
        ET15 = []
        OT15 = []
        ET16 = []
        OT16 = []
        #OV split with 1
        ind=-1
        for num in OV1:
            ind+=1
            if ind % 2 == 0:
                ET1.append(num)
            elif ind % 2 != 0:
                OT1.append(num)
        #EV split with 1
        ind=-1
        for num in EV1:
            ind+=1
            if ind % 2 == 0:
                ET2.append(num)
            elif ind % 2 != 0:
                OT2.append(num)
        #OV split with 2
        ind=-1
        for num in OV2:
            ind+=1
            if ind % 2 == 0:
                ET3.append(num)
            elif ind % 2 != 0:
                OT3.append(num)
        #EV split with 2
        ind=-1
        for num in EV2:
            ind+=1
            if ind % 2 == 0:
                ET4.append(num)
            elif ind % 2 != 0:
                OT4.append(num)
        #OV split with 3
        ind=-1
        for num in OV3:
            ind+=1
            if ind % 2 == 0:
                ET5.append(num)
            elif ind % 2 != 0:
                OT5.append(num)
        #EV split with 3
        ind=-1
        for num in EV3:
            ind+=1
            if ind % 2 == 0:
                ET6.append(num)
            elif ind % 2 != 0:
                OT6.append(num)
        #OV split with 4
        ind=-1
        for num in OV4:
            ind+=1
            if ind % 2 == 0:
                ET7.append(num)
            elif ind % 2 != 0:
                OT7.append(num)
        #EV split with 4
        ind=-1
        for num in EV4:
            ind+=1
            if ind % 2 == 0:
                ET8.append(num)
            elif ind % 2 != 0:
                OT8.append(num)
        #OV split with 5
        ind=-1
        for num in OV5:
            ind+=1
            if ind % 2 == 0:
                ET9.append(num)
            elif ind % 2 != 0:
                OT9.append(num)
        #EV split with 5
        ind=-1
        for num in EV5:
            ind+=1
            if ind % 2 == 0:
                ET10.append(num)
            elif ind % 2 != 0:
                OT10.append(num)
        #OV split with 6
        ind=-1
        for num in OV6:
            ind+=1
            if ind % 2 == 0:
                ET11.append(num)
            elif ind % 2 != 0:
                OT11.append(num)
        #EV split with 6
        ind=-1
        for num in EV6:
            ind+=1
            if ind % 2 == 0:
                ET12.append(num)
            elif ind % 2 != 0:
                OT12.append(num)
        #OV split with 7
        ind=-1
        for num in OV7:
            ind+=1
            if ind % 2 == 0:
                ET13.append(num)
            elif ind % 2 != 0:
                OT13.append(num)
        #EV split with 7
        ind=-1
        for num in EV7:
            ind+=1
            if ind % 2 == 0:
                ET14.append(num)
            elif ind % 2 != 0:
                OT14.append(num)
        #OV split with 8
        ind=-1
        for num in OV8:
            ind+=1
            if ind % 2 == 0:
                ET15.append(num)
            elif ind % 2 != 0:
                OT15.append(num)
        #EV split with 8
        ind=-1
        for num in EV8:
            ind+=1
            if ind % 2 == 0:
                ET16.append(num)
            elif ind % 2 != 0:
                OT16.append(num)
        #Now Have ET,OT numbers 1-16 for 32 lists total only list too large if looking at extreme textures... outside
        
        
        #If Looking at extreme Textures
        if len(ET1) > 1884 and len(OT16) > 1884:
            EL1 = []
            OL1 = []
            EL2 = []
            OL2 = []
            EL3 = []
            OL3 = []
            EL4 = []
            OL4 = []
            EL5 = []
            OL5 = []
            EL6 = []
            OL6 = []
            EL7 = []
            OL7 = []
            EL8 = []
            OL8 = []
            EL9 = []
            OL9 = []
            EL10 = []
            OL10 = []
            EL11 = []
            OL11 = []
            EL12 = []
            OL12 = []
            EL13 = []
            OL13 = []
            EL14 = []
            OL14 = []
            EL15 = []
            OL15 = []
            EL16 = []
            OL16 = []
            EL17 = []
            OL17 = []
            EL18 = []
            OL18 = []
            EL19 = []
            OL19 = []
            EL20 = []
            OL20 = []
            EL21 = []
            OL21 = []
            EL22 = []
            OL22 = []
            EL23 = []
            OL23 = []
            EL24 = []
            OL24 = []
            EL25 = []
            OL25 = []
            EL26 = []
            OL26 = []
            EL27 = []
            OL27 = []
            EL28 = []
            OL28 = []
            EL29 = []
            OL29 = []
            EL30 = []
            OL30 = []
            EL31 = []
            OL31 = []
            EL32 = []
            OL32 = []
            #OT split with 1
            ind=-1
            for num in OT1:
                ind+=1
                if ind % 2 == 0:
                    EL1.append(num)
                elif ind % 2 != 0:
                    OL1.append(num)
            #ET split with 1
            ind=-1
            for num in ET1:
                ind+=1
                if ind % 2 == 0:
                    EL2.append(num)
                elif ind % 2 != 0:
                    OL2.append(num)
            #OT split with 2
            ind=-1
            for num in OT2:
                ind+=1
                if ind % 2 == 0:
                    EL3.append(num)
                elif ind % 2 != 0:
                    OL3.append(num)
            #ET split with 2
            ind=-1
            for num in ET2:
                ind+=1
                if ind % 2 == 0:
                    EL4.append(num)
                elif ind % 2 != 0:
                    OL4.append(num)
            #OT split with 3
            ind=-1
            for num in OT3:
                ind+=1
                if ind % 2 == 0:
                    EL5.append(num)
                elif ind % 2 != 0:
                    OL5.append(num)
            #ET split with 3
            ind=-1
            for num in ET3:
                ind+=1
                if ind % 2 == 0:
                    EL6.append(num)
                elif ind % 2 != 0:
                    OL6.append(num)
            #OT split with 4
            ind=-1
            for num in OT4:
                ind+=1
                if ind % 2 == 0:
                    EL7.append(num)
                elif ind % 2 != 0:
                    OL7.append(num)
            #ET split with 4
            ind=-1
            for num in ET4:
                ind+=1
                if ind % 2 == 0:
                    EL8.append(num)
                elif ind % 2 != 0:
                    OL8.append(num)
            #OT split with 5
            ind=-1
            for num in OT5:
                ind+=1
                if ind % 2 == 0:
                    EL9.append(num)
                elif ind % 2 != 0:
                    OL9.append(num)
            #ET split with 5
            ind=-1
            for num in ET5:
                ind+=1
                if ind % 2 == 0:
                    EL10.append(num)
                elif ind % 2 != 0:
                    OL10.append(num)
            #OT split with 6
            ind=-1
            for num in OT6:
                ind+=1
                if ind % 2 == 0:
                    EL11.append(num)
                elif ind % 2 != 0:
                    OL11.append(num)
            #ET split with 6
            ind=-1
            for num in ET6:
                ind+=1
                if ind % 2 == 0:
                    EL12.append(num)
                elif ind % 2 != 0:
                    OL12.append(num)
            #OT split with 7
            ind=-1
            for num in OT7:
                ind+=1
                if ind % 2 == 0:
                    EL13.append(num)
                elif ind % 2 != 0:
                    OL13.append(num)
            #ET split with 7
            ind=-1
            for num in ET7:
                ind+=1
                if ind % 2 == 0:
                    EL14.append(num)
                elif ind % 2 != 0:
                    OL14.append(num)
            #OT split with 8
            ind=-1
            for num in OT8:
                ind+=1
                if ind % 2 == 0:
                    EL15.append(num)
                elif ind % 2 != 0:
                    OL15.append(num)
            #ET split with 8
            ind=-1
            for num in ET8:
                ind+=1
                if ind % 2 == 0:
                    EL16.append(num)
                elif ind % 2 != 0:
                    OL16.append(num)
            #OT split with 9
            ind=-1
            for num in OT9:
                ind+=1
                if ind % 2 == 0:
                    EL17.append(num)
                elif ind % 2 != 0:
                    OL17.append(num)
            #ET split with 9
            ind=-1
            for num in ET9:
                ind+=1
                if ind % 2 == 0:
                    EL18.append(num)
                elif ind % 2 != 0:
                    OL18.append(num)
            #OT split with 10
            ind=-1
            for num in OT10:
                ind+=1
                if ind % 2 == 0:
                    EL19.append(num)
                elif ind % 2 != 0:
                    OL19.append(num)
            #ET split with 10
            ind=-1
            for num in ET10:
                ind+=1
                if ind % 2 == 0:
                    EL20.append(num)
                elif ind % 2 != 0:
                    OL20.append(num)
            #OT split with 11
            ind=-1
            for num in OT11:
                ind+=1
                if ind % 2 == 0:
                    EL21.append(num)
                elif ind % 2 != 0:
                    OL21.append(num)
            #ET split with 11
            ind=-1
            for num in ET11:
                ind+=1
                if ind % 2 == 0:
                    EL22.append(num)
                elif ind % 2 != 0:
                    OL22.append(num)
            #OT split with 12
            ind=-1
            for num in OT12:
                ind+=1
                if ind % 2 == 0:
                    EL23.append(num)
                elif ind % 2 != 0:
                    OL23.append(num)
            #ET split with 12
            ind=-1
            for num in ET12:
                ind+=1
                if ind % 2 == 0:
                    EL24.append(num)
                elif ind % 2 != 0:
                    OL24.append(num)
            #OT split with 13
            ind=-1
            for num in OT13:
                ind+=1
                if ind % 2 == 0:
                    EL25.append(num)
                elif ind % 2 != 0:
                    OL25.append(num)
            #ET split with 13
            ind=-1
            for num in ET13:
                ind+=1
                if ind % 2 == 0:
                    EL26.append(num)
                elif ind % 2 != 0:
                    OL26.append(num)
            #OT split with 14
            ind=-1
            for num in OT14:
                ind+=1
                if ind % 2 == 0:
                    EL27.append(num)
                elif ind % 2 != 0:
                    OL27.append(num)
            #ET split with 14
            ind=-1
            for num in ET14:
                ind+=1
                if ind % 2 == 0:
                    EL28.append(num)
                elif ind % 2 != 0:
                    OL28.append(num)
            #OT split with 15
            ind=-1
            for num in OT15:
                ind+=1
                if ind % 2 == 0:
                    EL29.append(num)
                elif ind % 2 != 0:
                    OL29.append(num)
            #ET split with 15
            ind=-1
            for num in ET15:
                ind+=1
                if ind % 2 == 0:
                    EL30.append(num)
                elif ind % 2 != 0:
                    OL30.append(num)
            #OT split with 16
            ind=-1
            for num in OT16:
                ind+=1
                if ind % 2 == 0:
                    EL31.append(num)
                elif ind % 2 != 0:
                    OL31.append(num)
            #ET split with 16
            ind=-1
            for num in ET16:
                ind+=1
                if ind % 2 == 0:
                    EL32.append(num)
                elif ind % 2 != 0:
                    OL32.append(num)
            
            
            
            

    if len(EV1) <= 1884 and len(OV8) > 1 and len(EV4) <= 1884 and len(OV1) <= 1884 and len(OV4) <= 1884:
        #print('Small')
        Total = len(EV1)+len(OV1)+len(EV2)+len(OV2)+len(EV3)+len(OV3)+len(EV4)+len(OV4)+len(EV5)+len(OV5)+len(EV6)+len(OV6)+len(EV7)+len(OV7)+len(EV8)+len(OV8)
        #print(Total, Total/2)
        #First Groups
        O1A = []
        O1B = []
        E1A = []
        E1B = []
        #Seconds Groups
        O2A = []
        O2B = []
        O2C = []
        O2D = []
        E2A = []
        E2B = []
        E2C = []
        E2D = []
        #Third Groups
        O3A = []
        O3B = []
        O3C = []
        O3D = []
        O3E = []
        O3F = []
        O3G = []
        O3H = []
        E3A = []
        E3B = []
        E3C = []
        E3D = []
        E3E = []
        E3F = []
        E3G = []
        E3H = []
        ###############################
        if Total <= 7536 and Total > 0:
            O1A = []
            val = 0
            while val < 1884:
                if val % 4 == 0:
                    for num in EV1:
                        O1A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 1:
                    for num in EV2:
                        O1A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 2:
                    for num in EV3:
                        O1A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 3:
                    for num in EV4:
                        O1A.append(num)
                        val+=1
                        if val == 1884:
                            break
            E1A = []
            val = 0
            while val < 1884:
                if val % 4 == 0:
                    for num in EV5:
                        E1A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 1:
                    for num in EV6:
                        E1A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 2:
                    for num in EV7:
                        E1A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 3:
                    for num in EV8:
                        E1A.append(num)
                        val+=1
                        if val == 1884:
                            break
            O1B = []
            val = 0
            while val < 1884:
                if val % 4 == 0:
                    for num in OV1:
                        O1B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 1:
                    for num in OV2:
                        O1B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 2:
                    for num in OV3:
                        O1B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 3:
                    for num in OV4:
                        O1B.append(num)
                        val+=1
                        if val == 1884:
                            break
            E1B = []
            val = 0
            while val < 1884:
                if val % 4 == 0:
                    for num in OV5:
                        E1B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 1:
                    for num in OV6:
                        E1B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 2:
                    for num in OV7:
                        E1B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 4 == 3:
                    for num in OV8:
                        E1B.append(num)
                        val+=1
                        if val == 1884:
                            break
            #print(len(O1A),len(O1B),len(E1A),len(E1B))
            E1 = []
            O1 = []
            E1.append(E1A)
            E1.append(E1B)
            O1.append(O1A)
            O1.append(O1B)
            #print(len(E1),len(O1))
            #Each length of 2
            return E1, O1
        #####################################
        elif Total <= 15072 and Total > 7536:
            O2A = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in EV1:
                        O2A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in EV2:
                        O2A.append(num)
                        val+=1
                        if val == 1884:
                            break
            O2B = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in EV3:
                        O2B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in EV4:
                        O2B.append(num)
                        val+=1
                        if val == 1884:
                            break
            E2A = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in EV5:
                        E2A.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in EV6:
                        E2A.append(num)
                        val+=1
                        if val == 1884:
                            break
            E2B = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in EV7:
                        E2B.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in EV8:
                        E2B.append(num)
                        val+=1
                        if val == 1884:
                            break
            O2C = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in OV1:
                        O2C.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in OV2:
                        O2C.append(num)
                        val+=1
                        if val == 1884:
                            break
            O2D = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in OV3:
                        O2D.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in OV4:
                        O2D.append(num)
                        val+=1
                        if val == 1884:
                            break
            E2C = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in OV5:
                        E2C.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in OV6:
                        E2C.append(num)
                        val+=1
                        if val == 1884:
                            break
            E2D = []
            val = 0
            while val < 1884:
                if val % 2 == 0:
                    for num in OV7:
                        E2D.append(num)
                        val+=1
                        if val == 1884:
                            break
                elif val % 2 == 1:
                    for num in OV8:
                        E2D.append(num)
                        val+=1
                        if val == 1884:
                            break
            #print(len(O2A),len(O2B),len(O2C),len(O2D),len(E2A),len(E2B),len(E2C),len(E2D))
            E2=[]
            O2=[]
            E2.append(E2A)
            E2.append(E2B)
            E2.append(E2C)
            E2.append(E2D)
            O2.append(O2A)
            O2.append(O2B)
            O2.append(O2C)
            O2.append(O2D)
            #print(len(O2),len(E2))
            #Each len of 4
            return E2, O2
        ######################################
        elif Total <= 30144 and Total > 15072:
            O3A = []
            val = 0
            while val < 1884:
                for num in EV1:
                    O3A.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3B = []
            val = 0
            while val < 1884:
                for num in EV2:
                    O3B.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3C = []
            val = 0
            while val < 1884:
                for num in EV3:
                    O3C.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3D = []
            val = 0
            while val < 1884:
                for num in EV4:
                    O3D.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3A = []
            val = 0
            while val < 1884:
                for num in EV5:
                    E3A.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3B = []
            val = 0
            while val < 1884:
                for num in EV6:
                    E3B.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3C = []
            val = 0
            while val < 1884:
                for num in EV7:
                    E3C.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3D = []
            val = 0
            while val < 1884:
                for num in EV8:
                    E3D.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3E = []
            val = 0
            while val < 1884:
                for num in OV1:
                    O3E.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3F = []
            val = 0
            while val < 1884:
                for num in OV2:
                    O3F.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3G = []
            val = 0
            while val < 1884:
                for num in OV3:
                    O3G.append(num)
                    val+=1
                    if val == 1884:
                        break
            O3H = []
            val = 0
            while val < 1884:
                for num in OV4:
                    O3H.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3E = []
            val = 0
            while val < 1884:
                for num in OV5:
                    E3E.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3F = []
            val = 0
            while val < 1884:
                for num in OV6:
                    E3F.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3G = []
            val = 0
            while val < 1884:
                for num in OV7:
                    E3G.append(num)
                    val+=1
                    if val == 1884:
                        break
            E3H = []
            val = 0
            while val < 1884:
                for num in OV8:
                    E3H.append(num)
                    val+=1
                    if val == 1884:
                        break
            #print(len(O3A),len(O3B),len(O3C),len(O3D),len(O3E),len(O3F),len(O3G),len(O3H),len(E3A),len(E3B),len(E3C),len(E3D),len(E3E),len(E3F),len(E3G),len(E3H))
            E3=[]
            O3=[]
            E3.append(E3A)
            E3.append(E3B)
            E3.append(E3C)
            E3.append(E3D)
            E3.append(E3E)
            E3.append(E3F)
            E3.append(E3G)
            E3.append(E3H)
            O3.append(O3A)
            O3.append(O3B)
            O3.append(O3C)
            O3.append(O3D)
            O3.append(O3E)
            O3.append(O3F)
            O3.append(O3G)
            O3.append(O3H)
            #print(len(O3),len(E3))
            #Each length of 8
            return E3, O3
            
            
            
            
        #Has Potentialities for manipulations of E/O (V) 1-8
        #Need to create all 1884 len lists
        
        #print(len(EV1),len(OV8))
        
    elif len(ET1) <= 1884 and len(OT16) > 1 and len(ET8) <= 1884 and len(OT1) <= 1884 and len(OT8) <= 1884:
        ######################################################################################################
        #print('Large')
        TotalE = len(ET1)+len(ET2)+len(ET3)+len(ET4)+len(ET5)+len(ET6)+len(ET7)+len(ET8)+len(ET9)+len(ET10)+len(ET11)+len(ET12)+len(ET13)+len(ET14)+len(ET15)+len(ET16)
        TotalO = len(OT1)+len(OT2)+len(OT3)+len(OT4)+len(OT5)+len(OT6)+len(OT7)+len(OT8)+len(OT9)+len(OT10)+len(OT11)+len(OT12)+len(OT13)+len(OT14)+len(OT15)+len(OT16)
        Total = TotalE + TotalO
        #print(Total, Total/2)
        #Grouping
        O41A=[]
        O41B=[]
        O41C=[]
        O41D=[]
        O41E=[]
        O41F=[]
        O41G=[]
        O41H=[]
        E41A=[]
        E41B=[]
        E41C=[]
        E41D=[]
        E41E=[]
        E41F=[]
        E41G=[]
        E41H=[]
        O42A=[]
        O42B=[]
        O42C=[]
        O42D=[]
        O42E=[]
        O42F=[]
        O42G=[]
        O42H=[]
        E42A=[]
        E42B=[]
        E42C=[]
        E42D=[]
        E42E=[]
        E42F=[]
        E42G=[]
        E42H=[]
        if Total > 30144 and Total < 60288:
            O41A=[]
            val=0
            while val<1884:
                for num in ET1:
                    O41A.append(num)
                    val+=1
                    if val==1884:
                        break
            O41B=[]
            val=0
            while val<1884:
                for num in ET2:
                    O41B.append(num)
                    val+=1
                    if val==1884:
                        break
            O41C=[]
            val=0
            while val<1884:
                for num in ET3:
                    O41C.append(num)
                    val+=1
                    if val==1884:
                        break
            O41D=[]
            val=0
            while val<1884:
                for num in ET4:
                    O41D.append(num)
                    val+=1
                    if val<1884:
                        break
            O41E=[]
            val=0
            while val<1884:
                for num in ET5:
                    O41E.append(num)
                    val+=1
                    if val==1884:
                        break
            O41F=[]
            val=0
            while val<1884:
                for num in ET6:
                    O41F.append(num)
                    val+=1
                    if val==1884:
                        break
            O41G=[]
            val=0
            while val<1884:
                for num in ET7:
                    O41G.append(num)
                    val+=1
                    if val==1884:
                        break
            O41H=[]
            val=0
            while val<1884:
                for num in ET8:
                    O41H.append(num)
                    val+=1
                    if val==1884:
                        break
            E41A=[]
            val=0
            while val<1884:
                for num in ET9:
                    E41A.append(num)
                    val+=1
                    if val==1884:
                        break
            E41B=[]
            val=0
            while val<1884:
                for num in ET10:
                    E41B.append(num)
                    val+=1
                    if val==1884:
                        break
            E41C=[]
            val=0
            while val<1884:
                for num in ET11:
                    E41C.append(num)
                    val+=1
                    if val==1884:
                        break
            E41D=[]
            val=0
            while val<1884:
                for num in ET12:
                    E41D.append(num)
                    val+=1
                    if val ==1884:
                        break
            E41E=[]
            val=0
            while val<1884:
                for num in ET13:
                    E41E.append(num)
                    val+=1
                    if val==1884:
                        break
            E41F=[]
            val=0
            while val<1884:
                for num in ET14:
                    E41G.append(num)
                    val+=1
                    if val==1884:
                        break
            E41G=[]
            val=0
            while val<1884:
                for num in ET15:
                    E41G.append(num)
                    val+=1
                    if val==1884:
                        break
            E41H=[]
            val=0
            while val<1884:
                for num in ET16:
                    E41H.append(num)
                    val+=1
                    if val==1884:
                        break
            O42A=[]
            val=0
            while val<1884:
                for num in OT1:
                    O42A.append(num)
                    val+=1
                    if val==1884:
                        break
            O42B=[]
            val=0
            while val<1884:
                for num in OT2:
                    O42B.append(num)
                    val+=1
                    if val==1884:
                        break
            O42C=[]
            val=0
            while val<1884:
                for num in OT3:
                    O42C.append(num)
                    val+=1
                    if val==1884:
                        break
            O42D=[]
            val=0
            while val<1884:
                for num in OT4:
                    O42D.append(num)
                    val+=1
                    if val==1884:
                        break
            O42E=[]
            val=0
            while val<1884:
                for num in OT5:
                    O42E.append(num)
                    val+=1
                    if val==1884:
                        break
            O42F=[]
            val=0
            while val<1884:
                for num in OT6:
                    O42F.append(num)
                    val+=1
                    if val==1884:
                        break
            O42G=[]
            val=0
            while val<1884:
                for num in OT7:
                    O42G.append(num)
                    val+=1
                    if val==1884:
                        break
            O42H=[]
            val=0
            while val<1884:
                for num in OT8:
                    O42H.append(num)
                    val+=1
                    if val==1884:
                        break
            E42A=[]
            val=0
            while val<1884:
                for num in OT9:
                    E42A.append(num)
                    val+=1
                    if val==1884:
                        break
            E42B=[]
            val=0
            while val<1884:
                for num in OT10:
                    E42B.append(num)
                    val+=1
                    if val==1884:
                        break
            E42C=[]
            val=0
            while val<1884:
                for num in OT11:
                    E42C.append(num)
                    val+=1
                    if val==1884:
                        break
            E42D=[]
            val=0
            while val<1884:
                for num in OT12:
                    E42D.append(num)
                    val+=1
                    if val==1884:
                        break
            E42E=[]
            val=0
            while val<1884:
                for num in OT13:
                    E42E.append(num)
                    val+=1
                    if val==1884:
                        break
            E42F=[]
            val=0
            while val<1884:
                for num in OT14:
                    E42F.append(num)
                    val+=1
                    if val==1884:
                        break
            E42G=[]
            val=0
            while val<1884:
                for num in OT15:
                    E42G.append(num)
                    val+=1
                    if val==1884:
                        break
            E42H=[]
            val=0
            while val<1884:
                for num in OT16:
                    E42H.append(num)
                    val+=1
                    if val==1884:
                        break
            #print(len(O41A),len(O41H),len(E41A),len(E41H),len(O42A),len(O42H),len(E42A),len(E42H))
            E4=[]
            O4=[]
            E4.append(E41A)
            E4.append(E41B)
            E4.append(E41C)
            E4.append(E41D)
            E4.append(E41E)
            E4.append(E41F)
            E4.append(E41G)
            E4.append(E41H)
            E4.append(E42A)
            E4.append(E42B)
            E4.append(E42C)
            E4.append(E42D)
            E4.append(E42E)
            E4.append(E42F)
            E4.append(E42G)
            E4.append(E42H)
            #
            O4.append(O41A)
            O4.append(O41B)
            O4.append(O41C)
            O4.append(O41D)
            O4.append(O41E)
            O4.append(O41F)
            O4.append(O41G)
            O4.append(O41H)
            O4.append(O42A)
            O4.append(O42B)
            O4.append(O42C)
            O4.append(O42D)
            O4.append(O42E)
            O4.append(O42F)
            O4.append(O42G)
            O4.append(O42H)
            #print(len(E4),len(O4))
            #Each has length of 16
            return E4, O4
            
        #Has Potentialities for manipulations of E/O (T) 1-16
        #Need to create all 1884 len lists
    ######################################################################################
    elif len(EL1) <= 1884 and len(OL32) > 1 and len(EL16) <= 1884:
        #Has Potentialities for manipulations of E/O (L) 1-32
        #Need to create all 1884 len lists
        #print('Biggest Split')
        Total1 = len(EL1)+len(EL2)+len(EL3)+len(EL4)+len(EL5)+len(EL6)+len(EL7)+len(EL8)+len(EL9)+len(EL10)+len(EL11)+len(EL12)+len(EL13)+len(EL14)+len(EL15)+len(EL16)
        Total2 = len(OL1)+len(OL2)+len(OL3)+len(OL4)+len(OL5)+len(OL6)+len(OL7)+len(OL8)+len(OL9)+len(OL10)+len(OL11)+len(OL12)+len(OL13)+len(OL14)+len(OL15)+len(OL16)
        Total3 = len(EL17)+len(EL18)+len(EL19)+len(EL20)+len(EL21)+len(EL22)+len(EL23)+len(EL24)+len(EL25)+len(EL26)+len(EL27)+len(EL28)+len(EL29)+len(EL30)+len(EL31)+len(EL32)
        Total4 = len(OL17)+len(OL18)+len(OL19)+len(OL20)+len(OL21)+len(OL22)+len(OL23)+len(OL24)+len(OL25)+len(OL26)+len(OL27)+len(OL28)+len(OL29)+len(OL30)+len(OL31)+len(OL32)
        Total = Total1+Total2+Total3+Total4
        #print(Total)
        O51A=[]
        O51B=[]
        O51C=[]
        O51D=[]
        O51E=[]
        O51F=[]
        O51G=[]
        O51H=[]
        O51I=[]
        O51J=[]
        O51K=[]
        O51L=[]
        O51M=[]
        O51N=[]
        O51O=[]
        O51P=[]
        E51A=[]
        E51B=[]
        E51C=[]
        E51D=[]
        E51E=[]
        E51F=[]
        E51G=[]
        E51H=[]
        E51I=[]
        E51J=[]
        E51K=[]
        E51L=[]
        E51M=[]
        E51N=[]
        E51O=[]
        E51P=[]
        O52A=[]
        O52B=[]
        O52C=[]
        O52D=[]
        O52E=[]
        O52F=[]
        O52G=[]
        O52H=[]
        O52I=[]
        O52J=[]
        O52K=[]
        O52L=[]
        O52M=[]
        O52N=[]
        O52O=[]
        O52P=[]
        E52A=[]
        E52B=[]
        E52C=[]
        E52D=[]
        E52E=[]
        E52F=[]
        E52G=[]
        E52H=[]
        E52I=[]
        E52J=[]
        E52K=[]
        E52L=[]
        E52M=[]
        E52N=[]
        E52O=[]
        E52P=[]
        if Total > 60288:
            O51A=[]
            val=0
            while val<1884:
                for num in EL1:
                    O51A.append(num)
                    val+=1
                    if val==1884:
                        break
            O51B=[]
            val=0
            while val<1884:
                for num in EL2:
                    O51B.append(num)
                    val+=1
                    if val==1884:
                        break
            O51C=[]
            val=0
            while val<1884:
                for num in EL3:
                    O51C.append(num)
                    val+=1
                    if val==1884:
                        break
            O51D=[]
            val=0
            while val<1884:
                for num in EL4:
                    O51D.append(num)
                    val+=1
                    if val==1884:
                        break
            O51E=[]
            val=0
            while val<1884:
                for num in EL5:
                    O51E.append(num)
                    val+=1
                    if val==1884:
                        break
            O51F=[]
            val=0
            while val<1884:
                for num in EL6:
                    O51F.append(num)
                    val+=1
                    if val==1884:
                        break
            O51G=[]
            val=0
            while val<1884:
                for num in EL7:
                    O51G.append(num)
                    val+=1
                    if val==1884:
                        break
            O51H=[]
            val=0
            while val<1884:
                for num in EL8:
                    O51H.append(num)
                    val+=1
                    if val==1884:
                        break
            O51I=[]
            val=0
            while val<1884:
                for num in EL9:
                    O51I.append(num)
                    val+=1
                    if val==1884:
                        break
            O51J=[]
            val=0
            while val<1884:
                for num in EL10:
                    O51J.append(num)
                    val+=1
                    if val==1884:
                        break
            O51K=[]
            val=0
            while val<1884:
                for num in EL11:
                    O51K.append(num)
                    val+=1
                    if val==1884:
                        break
            O51L=[]
            val=0
            while val<1884:
                for num in EL12:
                    O51L.append(num)
                    val+=1
                    if val==1884:
                        break
            O51M=[]
            val=0
            while val<1884:
                for num in EL13:
                    O51M.append(num)
                    val+=1
                    if val==1884:
                        break
            O51N=[]
            val=0
            while val<1884:
                for num in EL14:
                    O51N.append(num)
                    val+=1
                    if val==1884:
                        break
            O51O=[]
            val=0
            while val<1884:
                for num in EL15:
                    O51O.append(num)
                    val+=1
                    if val==1884:
                        break
            O51P=[]
            val=0
            while val<1884:
                for num in EL16:
                    O51P.append(num)
                    val+=1
                    if val==1884:
                        break
            E51A=[]
            val=0
            while val<1884:
                for num in EL17:
                    E51A.append(num)
                    val+=1
                    if val==1884:
                        break
            E51B=[]
            val=0
            while val<1884:
                for num in EL18:
                    E51B.append(num)
                    val+=1
                    if val==1884:
                        break
            E51C=[]
            val=0
            while val<1884:
                for num in EL19:
                    E51C.append(num)
                    val+=1
                    if val==1884:
                        break
            E51D=[]
            val=0
            while val<1884:
                for num in EL20:
                    E51D.append(num)
                    val+=1
                    if val==1884:
                        break
            E51E=[]
            val=0
            while val<1884:
                for num in EL21:
                    E51E.append(num)
                    val+=1
                    if val==1884:
                        break
            E51F=[]
            val=0
            while val<1884:
                for num in EL22:
                    E51F.append(num)
                    val+=1
                    if val==1884:
                        break
            E51G=[]
            val=0
            while val<1884:
                for num in EL23:
                    E51G.append(num)
                    val+=1
                    if val==1884:
                        break
            E51H=[]
            val=0
            while val<1884:
                for num in EL24:
                    E51H.append(num)
                    val+=1
                    if val==1884:
                        break
            E51I=[]
            val=0
            while val<1884:
                for num in EL25:
                    E51I.append(num)
                    val+=1
                    if val==1884:
                        break
            E51J=[]
            val=0
            while val<1884:
                for num in EL26:
                    E51J.append(num)
                    val+=1
                    if val==1884:
                        break
            E51K=[]
            val=0
            while val<1884:
                for num in EL27:
                    E51K.append(num)
                    val+=1
                    if val==1884:
                        break
            E51L=[]
            val=0
            while val<1884:
                for num in EL28:
                    E51L.append(num)
                    val+=1
                    if val==1884:
                        break
            E51M=[]
            val=0
            while val<1884:
                for num in EL29:
                    E51M.append(num)
                    val+=1
                    if val==1884:
                        break
            E51N=[]
            val=0
            while val<1884:
                for num in EL30:
                    E51N.append(num)
                    val+=1
                    if val==1884:
                        break
            E51O=[]
            val=0
            while val<1884:
                for num in EL31:
                    E51O.append(num)
                    val+=1
                    if val==1884:
                        break
            E51P=[]
            val=0
            while val<1884:
                for num in EL32:
                    E51P.append(num)
                    val+=1
                    if val==1884:
                        break
            ##########################
            O52A=[]
            val=0
            while val<1884:
                for num in OL1:
                    O52A.append(num)
                    val+=1
                    if val==1884:
                        break
            O52B=[]
            val=0
            while val<1884:
                for num in OL2:
                    O52B.append(num)
                    val+=1
                    if val==1884:
                        break
            O52C=[]
            val=0
            while val<1884:
                for num in OL3:
                    O52C.append(num)
                    val+=1
                    if val==1884:
                        break
            O52D=[]
            val=0
            while val<1884:
                for num in OL4:
                    O52D.append(num)
                    val+=1
                    if val==1884:
                        break
            O52E=[]
            val=0
            while val<1884:
                for num in OL5:
                    O52E.append(num)
                    val+=1
                    if val==1884:
                        break
            O52F=[]
            val=0
            while val<1884:
                for num in OL6:
                    O52F.append(num)
                    val+=1
                    if val==1884:
                        break
            O52G=[]
            val=0
            while val<1884:
                for num in OL7:
                    O52G.append(num)
                    val+=1
                    if val==1884:
                        break
            O52H=[]
            val=0
            while val<1884:
                for num in OL8:
                    O52H.append(num)
                    val+=1
                    if val==1884:
                        break
            O52I=[]
            val=0
            while val<1884:
                for num in OL9:
                    O52I.append(num)
                    val+=1
                    if val==1884:
                        break
            O52J=[]
            val=0
            while val<1884:
                for num in OL10:
                    O52J.append(num)
                    val+=1
                    if val==1884:
                        break
            O52K=[]
            val=0
            while val<1884:
                for num in OL11:
                    O52K.append(num)
                    val+=1
                    if val==1884:
                        break
            O52L=[]
            val=0
            while val<1884:
                for num in OL12:
                    O52L.append(num)
                    val+=1
                    if val==1884:
                        break
            O52M=[]
            val=0
            while val<1884:
                for num in OL13:
                    O52M.append(num)
                    val+=1
                    if val==1884:
                        break
            O52N=[]
            val=0
            while val<1884:
                for num in OL14:
                    O52N.append(num)
                    val+=1
                    if val==1884:
                        break
            O52O=[]
            val=0
            while val<1884:
                for num in OL15:
                    O52O.append(num)
                    val+=1
                    if val==1884:
                        break
            O52P=[]
            val=0
            while val<1884:
                for num in OL16:
                    O52P.append(num)
                    val+=1
                    if val==1884:
                        break
            E52A=[]
            val=0
            while val<1884:
                for num in OL17:
                    E52A.append(num)
                    val+=1
                    if val==1884:
                        break
            E52B=[]
            val=0
            while val<1884:
                for num in OL18:
                    E52B.append(num)
                    val+=1
                    if val==1884:
                        break
            E52C=[]
            val=0
            while val<1884:
                for num in OL19:
                    E52C.append(num)
                    val+=1
                    if val==1884:
                        break
            E52D=[]
            val=0
            while val<1884:
                for num in OL20:
                    E52D.append(num)
                    val+=1
                    if val==1884:
                        break
            E52E=[]
            val=0
            while val<1884:
                for num in OL21:
                    E52E.append(num)
                    val+=1
                    if val==1884:
                        break
            E52F=[]
            val=0
            while val<1884:
                for num in OL22:
                    E52F.append(num)
                    val+=1
                    if val==1884:
                        break
            E52G=[]
            val=0
            while val<1884:
                for num in OL23:
                    E52G.append(num)
                    val+=1
                    if val==1884:
                        break
            E52H=[]
            val=0
            while val<1884:
                for num in OL24:
                    E52H.append(num)
                    val+=1
                    if val==1884:
                        break
            E52I=[]
            val=0
            while val<1884:
                for num in OL25:
                    E52I.append(num)
                    val+=1
                    if val==1884:
                        break
            E52J=[]
            val=0
            while val<1884:
                for num in OL26:
                    E52J.append(num)
                    val+=1
                    if val==1884:
                        break
            E52K=[]
            val=0
            while val<1884:
                for num in OL27:
                    E52K.append(num)
                    val+=1
                    if val==1884:
                        break
            E52L=[]
            val=0
            while val<1884:
                for num in OL28:
                    E52L.append(num)
                    val+=1
                    if val==1884:
                        break
            E52M=[]
            val=0
            while val<1884:
                for num in OL29:
                    E52M.append(num)
                    val+=1
                    if val==1884:
                        break
            E52N=[]
            val=0
            while val<1884:
                for num in OL30:
                    E52N.append(num)
                    val+=1
                    if val==1884:
                        break
            E52O=[]
            val=0
            while val<1884:
                for num in OL31:
                    E52O.append(num)
                    val+=1
                    if val==1884:
                        break
            E52P=[]
            val=0
            while val<1884:
                for num in OL32:
                    E52P.append(num)
                    val+=1
                    if val==1884:
                        break
            E5=[]
            O5=[]
            E5.append(E51A)
            E5.append(E51B)
            E5.append(E51C)
            E5.append(E51D)
            E5.append(E51E)
            E5.append(E51F)
            E5.append(E51G)
            E5.append(E51H)
            E5.append(E51I)
            E5.append(E51J)
            E5.append(E51K)
            E5.append(E51L)
            E5.append(E51M)
            E5.append(E51N)
            E5.append(E51O)
            E5.append(E51P)
            E5.append(E52A)
            E5.append(E52B)
            E5.append(E52C)
            E5.append(E52D)
            E5.append(E52E)
            E5.append(E52F)
            E5.append(E52G)
            E5.append(E52H)
            E5.append(E52I)
            E5.append(E52J)
            E5.append(E52K)
            E5.append(E52L)
            E5.append(E52M)
            E5.append(E52N)
            E5.append(E52O)
            E5.append(E52P)
            #
            O5.append(O51A)
            O5.append(O51B)
            O5.append(O51C)
            O5.append(O51D)
            O5.append(O51E)
            O5.append(O51F)
            O5.append(O51G)
            O5.append(O51H)
            O5.append(O51I)
            O5.append(O51J)
            O5.append(O51K)
            O5.append(O51L)
            O5.append(O51M)
            O5.append(O51N)
            O5.append(O51O)
            O5.append(O51P)
            O5.append(O52A)
            O5.append(O52B)
            O5.append(O52C)
            O5.append(O52D)
            O5.append(O52E)
            O5.append(O52F)
            O5.append(O52G)
            O5.append(O52H)
            O5.append(O52I)
            O5.append(O52J)
            O5.append(O52K)
            O5.append(O52L)
            O5.append(O52M)
            O5.append(O52N)
            O5.append(O52O)
            O5.append(O52P)
            #print(len(E5),len(O5))
            #Each has length of 32
            return E5, O5
            
            

####################################CONTOURS#EXTRACTIONS###############################################################################################################

def HierarchyStract(hierarchy):
    Hval = []
    for two in hierarchy:
        for one in two:
            for val in one:
                Hval.append(val)
    #print('Break')
    #print(len(Hval))
    One=[]
    Two=[]
    ind=-1
    for val in Hval:
        ind+=1
        if ind % 2 == 0:
            One.append(val)
        elif ind % 2 == 1:
            Two.append(val)
    #print(len(One))
    #print(One)
    #print(len(Two))
    #print(Two)
    if len(One)<=1884 and len(Two)<=1884 and len(One)>0 and len(Two)>0:
        #print('First')
        #print(len(Hval))
        Uno = []
        val = 0
        while val<1884:
            for num in One:
                Uno.append(num)
                val+=1
                if val==1884:
                    break
        Dos = []
        val=0
        while val<1884:
            for num in Two:
                Dos.append(num)
                val+=1
                if val==1884:
                    break
        #print(len(Uno),len(Dos))
        return Uno, Dos
    elif len(One)>1884 and len(Two)>1884 and len(One)<=3768 and len(Two)<=3768:
        #print('Second')
        OneHalf = len(One)//2
        TwoHalf = len(Two)//2
        One1 = One[:OneHalf]
        One2 = One[OneHalf:]
        Two1 = Two[:TwoHalf]
        Two2 = Two[TwoHalf:]
        #print(len(One1),len(One2))
        Uno1 = []
        val = 0
        while val<1884:
            for num in One1:
                Uno1.append(num)
                val+=1
                if val==1884:
                    break
        Uno2=[]
        val=0
        while val<1884:
            for num in One2:
                Uno2.append(num)
                val+=1
                if val==1884:
                    break
        Dos1=[]
        val=0
        while val<1884:
            for num in Two1:
                Dos1.append(num)
                val+=1
                if val==1884:
                    break
        Dos2=[]
        val=0
        while val<1884:
            for num in Two2:
                Dos2.append(num)
                val+=1
                if val==1884:
                    break
        #print(len(Uno1),len(Uno2),len(Dos1),len(Dos2))
        Uno=[]
        Uno.append(Uno1)
        Uno.append(Uno2)
        Dos=[]
        Dos.append(Dos1)
        Dos.append(Dos2)
        #print(len(Uno),len(Dos))
        return Uno, Dos
    elif len(One)>3768 and len(Two)>3768 and len(One)<=7536 and len(Two)<=7536:
        #print('Third')
        Fourth1 = len(One)//4
        Fourth2 = len(Two)//4
        One1 = One[:Fourth1]
        One2 = One[Fourth1:Fourth1*2]
        One3 = One[Fourth1*2:Fourth1*3]
        One4 = One[Fourth1*3:]
        Two1 = Two[:Fourth2]
        Two2 = Two[Fourth2:Fourth2*2]
        Two3 = Two[Fourth2*2:Fourth2*3]
        Two4 = Two[Fourth2*3:]
        #Large 8 way splits
        Uno1 = []
        val = 0
        while val<1884:
            for num in One1:
                Uno1.append(num)
                val+=1
                if val==1884:
                    break
        Uno2=[]
        val=0
        while val<1884:
            for num in One2:
                Uno2.append(num)
                val+=1
                if val==1884:
                    break
        Uno3=[]
        val=0
        while val<1884:
            for num in One3:
                Uno3.append(num)
                val+=1
                if val==1884:
                    break
        Uno4=[]
        val=0
        while val<1884:
            for num in One4:
                Uno4.append(num)
                val+=1
                if val==1884:
                    break
        Dos1=[]
        val=0
        while val<1884:
            for num in Two1:
                Dos1.append(num)
                val+=1
                if val==1884:
                    break
        Dos2=[]
        val=0
        while val<1884:
            for num in Two2:
                Dos2.append(num)
                val+=1
                if val==1884:
                    break
        Dos3=[]
        val=0
        while val<1884:
            for num in Two3:
                Dos3.append(num)
                val+=1
                if val==1884:
                    break
        Dos4=[]
        val=0
        while val<1884:
            for num in Two4:
                Dos4.append(num)
                val+=1
                if val==1884:
                    break
        Uno=[]
        Uno.append(Uno1)
        Uno.append(Uno2)
        Uno.append(Uno3)
        Uno.append(Uno4)
        Dos=[]
        Dos.append(Dos1)
        Dos.append(Dos2)
        Dos.append(Dos3)
        Dos.append(Dos4)
        #print(len(Uno),len(Dos))
        return Uno,Dos
        
        






def Questions(WH, WC, WF, WB, WSNEC, WHNEC, FaceV2, EyeV2, A10):
    if num in FaceV2 > 5:
        #Preliminary Creation
        what0 = [(WC_i - FaceV2_i) / (WSNEC_i + EyeV2_i) for WC_i, FaceV2_i, WSNEC_i, EyeV2_i in zip(WC, FaceV2, WSNEC, EyeV2)]
        when0 = [(WC_i / WF_i) - (FaceV2_i + EyeV2_i) for WC_i, WF_i, FaceV2_i, EyeV2_i in zip(WC, WF, FaceV2, EyeV2)]
        where0 = [(WF_i * FaceV2_i) / (WH_i * EyeV2_i) for WF_i, FaceV2_i, WH_i, EyeV2_i in zip(WF, FaceV2, WH, EyeV2)]
        why0 = [(WH_i * WC_i / WHNEC_i) + FaceV2_i - EyeV2_i for WH_i, WC_i, WHNEC_i, FaceV2_i, EyeV2_i in zip(WH, WC, WHNEC, FaceV2, EyeV2)]
        who0 = [(WB_i + FaceV2_i) * (WF_i + EyeV2_i) for WB_i, FaceV2_i, WF_i, EyeV2_i in zip(WB, FaceV2, WF, EyeV2)]
        #Sperical Creation
        what = [what0_i / A10_i for what0_i, A10_i in zip(what0, A10)]
        when = [when0_i / A10_i for when0_i, A10_i in zip(when0, A10)]
        where = [where0_i / A10_i for where0_i, A10_i in zip(where0, A10)]
        why = [why0_i / A10_i for why0_i, A10_i in zip(why0, A10)]
        who = [who0_i / A10_i for who0_i, A10_i in zip(who0, A10)]
    else:
        #Preliminary Creation
        what0 = [WC_i / WSNEC_i for WC_i, WSNEC_i in zip(WC, WSNEC)]
        when0 = [WC_i / WF_i for WC_i, WF_i in zip(WC, WF)]
        where0 = [WF_i / WH_i for WF_i, WH_i in zip(WF, WH)]
        why0 = [WH_i * WC_i / WHNEC_i for WH_i, WC_i, WHNEC_i in zip(WH, WC, WHNEC)]
        who0 = [WB_i * WF_i for WB_i, WF_i in zip(WB, WF)]
        #Spherical Creation
        what = [what0_i / A10_i for what0_i, A10_i in zip(what0, A10)]
        when = [when0_i / A10_i for when0_i, A10_i in zip(when0, A10)]
        where = [where0_i / A10_i for where0_i, A10_i in zip(where0, A10)]
        why = [why0_i / A10_i for why0_i, A10_i in zip(why0, A10)]
        who = [who0_i / A10_i for who0_i, A10_i in zip(who0, A10)]
    return what, when, where, why, who


def Construct(prim0, prim, what, when, where, why, who, Q1, Q2, OLD1, Word1):
    #val = 0
    T0 = []
    Quantified0 = []
    #Processing of All Previous Words
    for word in prim0:
        for letter in word:
            #print(len(letter))
            val = sum(letter)
            T0.append(val)
    spot = 0
    spread = []
    Cond = []
    for num in OLD1:
        aug = -1
        for let in itertools.islice(T0, spot, spot+num):
            aug += 1
            if aug == 0:
                spread.append(let)
            elif aug == 1:
                spread.append(let)
            elif aug == 2:
                spread.append(let)
            elif aug == 3:
                spread.append(let)
            elif aug == 4:
                spread.append(let)
            elif aug == 5:
                spread.append(let)
            elif aug == 6:
                spread.append(let)
            elif aug == 7:
                spread.append(let)
            elif aug == 8:
                spread.append(let)
            elif aug == 9:
                spread.append(let)
            elif aug == 10:
                spread.append(let)
            elif aug == 11:
                spread.append(let)
            elif aug == 12:
                spread.append(let)
            elif aug == 13:
                spread.append(let)
            elif aug == 14:
                spread.append(let)
            
            
        Cond.append(spread)
        spread = []
        spot += num
    I = 0
    P = 0
    for wor in Cond:
        if len(wor) == 1:
            P = sum(wor)
            Quantified0.append(P)
        else:
            I = sum(wor)
            Quantified0.append(I)
    Waste0 = len(Quantified0)
    #Singular Values for Prim0, OLD1 accomplished
    #Waste Processing Portionalities
    Pos1 = (why / who) * Waste0 - when
    Neg1 = (what / where) * Waste0 + when
    #First
    Pos3 = 0
    Neg3 = 0
    for num in Quantified0:
        Pos3 += num
        Neg3 -= num

    #Processing of Newly input Words
    if len(prim) > 0:
        pal = 0
        P0 = []
        for word in prim:
            for letter in word:
                pal = sum(letter)
                P0.append(pal)
        spot = 0
        spread = []
        Cond0 = []
        for num in Word1:
            aug = -1
            for let in itertools.islice(P0, spot, spot+num):
                aug += 1
                if aug == 0:
                    spread.append(let)
                elif aug == 1:
                    spread.append(let)
                elif aug == 2:
                    spread.append(let)
                elif aug == 3:
                    spread.append(let)
                elif aug == 4:
                    spread.append(let)
                elif aug == 5:
                    spread.append(let)
                elif aug == 6:
                    spread.append(let)
                elif aug == 7:
                    spread.append(let)
                elif aug == 8:
                    spread.append(let)
                elif aug == 9:
                    spread.append(let)
                elif aug == 10:
                    spread.append(let)
                elif aug == 11:
                    spread.append(let)
                elif aug == 12:
                    spread.append(let)
                elif aug == 13:
                    spread.append(let)
                elif aug == 14:
                    spread.append(let)
                
            
            Cond0.append(spread)
            spread = []
            spot += num
        Quantified1 = []
        W = 0
        E = 0
        for letter in Cond0:
            if len(letter) == 1:
                E = sum(letter)
                Quantified1.append(E)
            else:
                W = sum(letter)
                Quantified1.append(W)
        SubG = Waste0 / len(Quantified1)
        #SubG Processing  Portionalities
        Pos2 = (where / who) * SubG - when
        Neg2 = (what / why) * SubG + when
        Pos4 = 0
        Neg4 = 0
        for num in Quantified1:
            Pos4 += num
            Neg4 -= num
        
        XPP = Pos2 / Pos1
        XNP = Pos3 / Pos4
        YNN = Neg2 / Neg1
        YPN = Neg3 / Neg4
        XY1 = XNP + YNN
        XY2 = XPP + YPN
        if Q1 != 0 and Q2 != 0:
            Q01 = XY1 / Q1
            Q02 = XY2 / Q2
            return Q01, Q02, Quantified0, Quantified1
        else:
            return XY1, XY2, Quantified0, Quantified1
    YNN1 = Neg3 / Neg1
    XPP1 = Pos1 / Pos3
    YXN1 = Neg3 / Pos1
    YXN2 = Pos3 / Neg1
    XN1 = YNN1 * YXN1
    YN1 = XPP1 * YXN2
    if Q1 != 0 and Q2 != 0:
        Q001 = XN1 / Q1
        Q002 = YN1 / Q2
        Q001 = 0 - where - 1
        Q002 = 0 + where + 1
        return Q001, Q002, Quantified0
    else:
        XN1 = 0 - where - 1
        YN1 = 0 + where + 1
        return XN1, YN1, Quantified0
    #
    #

def Speak(Q1, Q2, words, WInput, Quantified0, Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, contours, hierarchy):
    Quantified2 = []
    #Variation Creation
    spot = -1
    valuation = []
    Same = []
    for word in WInput:
        spot = -1
        for old in words:
            spot+=1
            if word == old:
                valuation.append(Quantified0[spot])
                Same.append(words[spot])
                spot = -1
            elif spot == len(words)-1:
                New0, New1 = WordCode(word, Em6, Om6, Em2, Om2)
                New1 = []
                
                New1.append(len(New0))
                A10, WH1, WC1, WF1, WB1, WSNEC1, WHNEC1, prim1, seeQ = DataInt(Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, contours, hierarchy, New0, New1)
                N0 = []
                QN = []
                T0 = []
                
                #Processing of All Previous Words
                for word in prim1:
                    for letter in word:
                        #print(len(letter))
                        val = sum(letter)
                        T0.append(val)
                spot = 0
                spread = []
                Cond = []
                for num in New1:
                    aug = -1
                    for let in itertools.islice(T0, spot, spot+num):
                        aug += 1
                        if aug == 0:
                            spread.append(let)
                        elif aug == 1:
                            spread.append(let)
                        elif aug == 2:
                            spread.append(let)
                        elif aug == 3:
                            spread.append(let)
                        elif aug == 4:
                            spread.append(let)
                        elif aug == 5:
                            spread.append(let)
                        elif aug == 6:
                            spread.append(let)
                        elif aug == 7:
                            spread.append(let)
                        elif aug == 8:
                            spread.append(let)
                        elif aug == 9:
                            spread.append(let)
                        elif aug == 10:
                            spread.append(let)
                        elif aug == 11:
                            spread.append(let)
                        elif aug == 12:
                            spread.append(let)
                        elif aug == 13:
                            spread.append(let)
                        elif aug == 14:
                            spread.append(let)
                        
            
                    Cond.append(spread)
                    spread = []
                    spot += num
                I = 0
                for letter in Cond:
                    I = sum(letter)
                    valuation.append(I)
    #print(len(valuation))
    #print(Same)
    #print(len(Quantified0))
    #print(len(Quantified2))
    #print(Quantified2)
    #Adding new portions to history
    print('Val',len(valuation))
    print('Same',len(Same))
    print('WInput',len(WInput))
    
    for word1 in WInput:
        words.append(word1)
    WInput = []
    for one in valuation:
        Quantified0.append(one)
    valuation=[]
    #Splitting factors : Original Split by valued odds/evens
    Odd = 0
    Even = 0
    OddQ = []
    OddW = []
    EvenQ = []
    EvenW = []
    spot = 0
    print(len(words))
    print(len(Quantified0))
    while spot < len(words):
        for num in Quantified0:
            anum = int(num)
            #print(spot)
            if anum % 2 == 0:
                if spot == len(words):
                    break
                elif spot < len(words):
                    Even += 1
                    EvenQ.append(Quantified0[spot])
                    #print(spot)
                    EvenW.append(words[spot])
            elif anum % 2 == 1:
                if spot == len(words):
                    break
                elif spot < len(words):
                    Odd += 1
                    OddQ.append(Quantified0[spot])
                    #print(spot)
                    OddW.append(words[spot])
            spot+=1
            
    #print('Odd', Odd)
    #print('Even', Even)
    print('Odd Words Quants')
    print(OddQ)
    print('Even Words Quants')
    print(EvenQ)
    
    return WInput
                    


FaceV2 = []
NeF = []
FaceV1 = []
FaceV = []
emt = []
EyeV = []
EyeV1 = []
EyeV2 = []
emp = []
Q1 = 0
Q2 = 0
Word0 = []
Word1 = []
prim = []
WInput = []
words0 = words
Quantified0 = []
Quantified1 = []
#RUNNING PROGRAM
while True:
    ret, frame = video_capture.read()
    #Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces1 = face_cascade.detectMultiScale(gray, 1.3, 3)
    faces2 = eye_cascade.detectMultiScale(gray, 1.3, 3)
    #Object Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame, (7,7), 0)

    canny = cv2.Canny(blurred, 20, 60)
    contours, hierarchy= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    #Even, Odd = ContourStract(contours)
    #Uno, Dos = HierarchyStract(hierarchy)
    time.sleep(.001)
    OIO = []
    Td = []
    EO = []
    while len(OIO) < 1884 and len(Td) < 1884:
        OIO.append(len(contours))
        #Time Counter
        #print(len(OIO))
        delta = dt.datetime.now()-t
        if delta.seconds > 0:
            Td.append(delta.seconds)
        elif delta.seconds == 0:
            Td.append(.35)
        #print(len(Td))
        #Label Creator / Halfing system
    #print(OIO)
    for i in range(len(Td)):
        if i % 2 == 0:
            EO.append(0)
        elif i % 2 != 0:
            EO.append(1)
    
    
    if len(words) != len(words) + len(words0):
        #print('Preprocess1')
        OLD0, OLD1 = WordCode(words, Em6, Om6, Em2, Om2)
        A10, WH0, WC0, WF0, WB0, WSNEC0, WHNEC0, prim0, seeQ0 = DataInt(Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, contours, hierarchy, OLD0, OLD1)
        what, when, where, why, who = Questions(WH0, WC0, WF0, WB0, WSNEC0, WHNEC0, FaceV2, EyeV2, A10)
        #print('Preprocess2')
    
    #print(WH0)
    #IMPORTANT DATA SETS : FaceV2, EyeV2
    #Face Data
    checkf = isinstance(faces1, tuple)

    #Eye Data
    checke = isinstance(faces2, tuple)
    
    
 
    
    #While Face and Eyes are detected
    if checkf == False and checke == False:
        FaceV1 = FaceStract(faces1)
        EyeV1 = FaceStract(faces2)
        #Face
        for num in FaceV1:
            FaceV.append(num)
            if len(FaceV) % 35 == 0:
                emt = FaceV
                FaceV2.append(emt)
                FaceV = []
        #Eyes
        for num in EyeV1:
            EyeV.append(num)
            if len(EyeV) % 35 == 0:
                emp = EyeV
                EyeV2.append(emp)
                EyeV = []
        
        
        #Listen for input
        x = delta.seconds
        if x >= 35:
            t = dt.datetime.now()
        elif Q1 < where[x] and Q2 > where[x]:
            audio = capture()
            try:
                WInput = process_text(audio)
                words0 = WInput
                print(WInput)
                Word0, Word1 = WordCode(WInput, Em6, Om6, Em2, Om2)
                A10, WH, WC, WF, WB, WSNEC, WHNEC, prim, seeQ = DataInt(Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, contours, hierarchy, Word0, Word1)
                #print(len(prim))
                what, when, where, why, who = Questions(WH, WC, WF, WB, WSNEC, WHNEC, FaceV2, EyeV2, A10)
            except TypeError:
                #print('Type Nope')
                WInput = []
                #print(WInput)
                Word0 = []
                Word1 = []
                prim = []
                words0 = []
                #print(type(words0))
                #print(type(words))
                #print('Maybe')
                
            print(what[x], when[x], where[x], why[x], who[x])
            Q1 = where[x] * where[x]
            Q2 = where[x] * where[x]
            #Q1 = 0
            #Q2 = 0
            print(x)
            #print(len(FaceV2))
        #Looking
        elif Q1 < where[x]:
            Q1 -= what[x] * why[x]
            Q3 = Q1
            Q4 = Q2
            print('Q1', Q1)
            #print('Quant', Quantified1)
            #print(len(Quantified1))
            if len(Quantified0) == 0 and len(WInput) > 0:
                Q1, Q2, Quantified0, Quantified1 = Construct(prim0, prim, what[x], when[x], where[x], why[x], who[x], Q1, Q2, OLD1, Word1)
            if len(WInput) > 0:
                WInput = Speak(Q1, Q2, words, WInput, Quantified0, Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, contours, hierarchy)
            Q1 = Q3
            Q2 = Q4
        #Looking
        elif Q2 > where[x]:
            Q2 -= when[x] / who[x]
            Q3 = Q2
            Q4 = Q1
            print('Q2', Q2)
            #print('Quant', Quantified1)
            #print(len(Quantified1))
            if len(Quantified0) == 0 and len(WInput) > 0:
                Q1, Q2, Quantified0, Quantified1 = Construct(prim0, prim, what[x], when[x], where[x], why[x], who[x], Q1, Q2, OLD1, Word1)
            if len(WInput) > 0:
                WInput = Speak(Q1, Q2, words, WInput, Quantified0, Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, contours, hierarchy)
            Q2 = Q3
            Q1 = Q4
        #Process for construction
        elif Q1 > where[x] and Q2 < where[x]:
            if len(Word0) > 0:
                Q1, Q2, Quantified0, Quantified1 = Construct(prim0, prim, what[x], when[x], where[x], why[x], who[x], Q1, Q2, OLD1, Word1)
                for letter in Word0:
                    OLD0.append(letter)
                for num in Word1:
                    OLD1.append(num)
                for word in prim:
                    prim0.append(word)
                for word in WInput:
                    words.append(word)
                Word0 = []
                Word1 = []
                prim = []
                WInput = []
                print('yes')
            #print(len(prim0))
            #print(len(OLD0))
            #print('Q3', Q2)
            elif len(Word0) == 0:
                Q1, Q2, Quantified0 = Construct(prim0, prim, what[x], when[x], where[x], why[x], who[x], Q1, Q2, OLD1, Word1)
                #print(len(words))
                #print(words)
                
            
    
    
    #While not Detected
    '''
    if checkf == True and checke == True:
        print('no Face')
        audio = capture()
        try:
            WInput = process_text(audio)
            print(WInput)
            Word0, Word1 = WordCode(WInput, Em6, Om6, Em2, Om2)
            WH, WC, WF, WB, WSNEC, WHNEC, prim = DataInt(Oe, E7, Em6, Om6, Om2, Em2, Em67, Om61, Om21, Em27, OIO, Td, Word0, Word1)
        except TypeError:
            print('Type Nope')
    
        #print(len(Word0))
        #print(Word1)
   
        #print(FaceV2)

    
    

    '''
    OIO = []
    Td = []
    EO = []



