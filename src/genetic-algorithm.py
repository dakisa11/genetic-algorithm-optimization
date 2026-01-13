import sys, math, argparse
import numpy as np

class ANN():
    def __init__(self):
        self.ulazi = []
        self.izlazi = []
        self.header = []
        self.B = []
        self.W = []
        self.akcije = []
        self.slojevi = []
    
    def initArh(self, nn):
        self.slojevi = []
        self.akcije = []
        vrijednosti = nn.split("s")
        self.slojevi.append(len(self.header)-1)
        for sloj in vrijednosti:
            if sloj == "":
                continue
            self.slojevi.append(int(sloj))
            self.akcije.append("s")
        self.slojevi.append(1)
        self.akcije.append(None)
        self.arhitektura()

    def arhitektura(self):
        self.W = []
        self.B = []
        for j in range(1, len(self.slojevi)):
            dim_prethodnog = self.slojevi[j-1]
            dim_trenutnog = self.slojevi[j]
            w_sloj = np.random.randn(dim_prethodnog, dim_trenutnog) * 0.01
            b_sloj = np.random.randn(1, dim_trenutnog) * 0.01
            self.W.append(w_sloj)
            self.B.append(b_sloj)

    def initNM(self, trainUlaz):
        self.header = trainUlaz.split()[0].split(",")
        ulazi_lista = []
        izlazi_lista = []
        for redak in trainUlaz.split()[1:]:
            vrijednosti = list(map(float, redak.split(",")))
            ulazi_lista.append(vrijednosti[:-1])
            izlazi_lista.append([vrijednosti[-1]])
        self.ulazi = np.array(ulazi_lista)
        self.izlazi = np.array(izlazi_lista)
    
    def sigm(self, net):
        return 1/(1 + np.exp(-net))
    
    def izravnaj(self):
        vektor = []
        for sloj in self.W:
            vektor.append(sloj.flatten())
        for sloj in self.B:
            vektor.append(sloj.flatten())
        #print(np.concatenate(vektor))
        return np.concatenate(vektor)
    
    def postavi(self, izgladen):
        pokazivac = 0
        
        noviW = []
        for sloj in self.W:
            velicina = sloj.size
            novi = (izgladen[pokazivac:pokazivac+velicina].reshape(sloj.shape))
            noviW.append(novi)
            pokazivac += velicina
        self.W = noviW

        noviB = []
        for sloj in self.B:
            velicina = sloj.size
            novi = (izgladen[pokazivac:pokazivac+velicina].reshape(sloj.shape))
            noviB.append(novi)
            pokazivac += velicina
        self.B = noviB

    def unaprijedni(self, X=None):
        if X is None:
            a = self.ulazi
        else:
            a = np.array(X)
            if a.ndim == 1:
                a = a.reshape(1, -1)

        for i in range(len(self.W)):
            z = a.dot(self.W[i]) + self.B[i]
            if self.akcije[i] == "s":
                a = self.sigm(z)
            else:
                a = z
        return a
    
    def greska(self, X = None, y = None):
        if X is None:
            X = self.ulazi
        if y is None:
            y = self.izlazi

        predikcija = self.unaprijedni(X)        
        return np.mean((y - predikcija)**2)

class GenetskiAlgoritam():
    def __init__(self, popSize, elitizam, Pmutacija, K, iter, nn):
        self.popSize = popSize
        self.elitizam = elitizam
        self.Pmutacija = Pmutacija
        self.K = K
        self.iter = iter
        self.nn = nn

        self.populacija = []
        self.dobrote = []

    def initPop(self):
        self.populacija = []
        for _ in range(self.popSize):
            self.nn.arhitektura()
            jedinka = self.nn.izravnaj()
            self.populacija.append(jedinka)
        
    def evaluiraj(self):
        self.dobrote = []
        for jedinka in self.populacija:
            self.nn.postavi(jedinka)
            greska = self.nn.greska()
            dobrota = 1 / (1 + greska)
            self.dobrote.append(dobrota)

    def odaberiRoditelje(self):
        ukupna = sum(self.dobrote)
        if ukupna == 0:
            vjerojatnosti = [1/len(self.dobrote)] * len(self.dobrote)
        else:
            vjerojatnosti = [d/ukupna for d in self.dobrote]
        index1 = np.random.choice(len(self.populacija), p = vjerojatnosti)
        index2 = np.random.choice(len(self.populacija), p = vjerojatnosti)
        return self.populacija[index1], self.populacija[index2]

    def krizaj(self, roditelj1, roditelj2):
        dijete = (roditelj1 + roditelj2)/2
        return dijete
    
    def mutiraj(self, jedinka):
        maska = np.random.rand(len(jedinka)) < self.Pmutacija
        suma = np.random.normal(0, self.K, len(jedinka))
        jedinka[maska] += suma[maska]
        return jedinka
    
    def novaGeneracija(self):
        indeksi = np.argsort(self.dobrote)[::-1]
        nova = [self.populacija[i] for i in indeksi[:self.elitizam]]

        for _ in range(self.popSize - self.elitizam):
            roditelj1, roditelj2 = self.odaberiRoditelje()
            dijete = self.krizaj(roditelj1, roditelj2)
            dijete = self.mutiraj(dijete)
            nova.append(dijete)
        
        self.populacija = nova
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = str)
    parser.add_argument("--test", type = str)
    parser.add_argument("--nn", type = str)
    parser.add_argument("--popsize", type = int)
    parser.add_argument("--elitism", type = int)
    parser.add_argument("--p", type = float)
    parser.add_argument("--K", type = float)
    parser.add_argument("--iter", type = int)
    args = parser.parse_args()

    with open(args.train) as f:
        trainUlaz = f.read()
    with open(args.test) as f:
        testUlaz = f.read()

    NM = ANN()
    NM.initNM(trainUlaz)
    NM.initArh(args.nn)

    NM_test = ANN()
    NM_test.initNM(testUlaz)
    NM_test.initArh(args.nn)

    GA = GenetskiAlgoritam(
        popSize = args.popsize,
        elitizam = args.elitism,
        Pmutacija = args.p, 
        K = args.K,
        iter = args.iter,
        nn=NM
    )
    
    GA.initPop()

    najboljaGreska = float('inf')
    najboljaJedinka = None

    for generacija in range(1, args.iter + 1):
        GA.evaluiraj()

        trenutnoNajboljaGreska = min([NM.greska() for _  in GA.populacija])
        if trenutnoNajboljaGreska < najboljaGreska:
            najboljaGreska = trenutnoNajboljaGreska
            najboljaJedinka =  GA.populacija[np.argmax(GA.dobrote)].copy()

        if generacija % 2000 == 0:
            print(f"[Train error @{generacija}]: {najboljaGreska}")
        
        GA.novaGeneracija()

    if najboljaJedinka is not None:
        NM_test.postavi(najboljaJedinka)
        testnaGreska = NM_test.greska()
        print(f"[Test error]: {testnaGreska}")

if __name__ == "__main__":
    main()