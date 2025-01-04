from multiplicationTriple import mult_triples
import time
import os

if __name__ == "__main__":
    file_name = "offline_phase.txt"
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print("The file does not exist") 
    
    batch_size = 128
    n = [1000, 10000, 100000]
    d = [100, 500, 1000]
    l = 64 # page 13 of MZ17 assume that values of dataset are 64 bit decimal number
    
    for i in range(len(n)):
        for j in range(len(d)):
            f = open(file_name, "a")
            f.write(f"Experiment - n: {n[i]} - d: {d[j]} \n")
            print(f"Running experiment n={n[i]}, d={d[i]}...")
            start_time = time.time()
            t = n[i] // batch_size
            mult_triples(n=n[i], d=d[j], t=t, l=l, batch_size=batch_size)
            end_time = time.time()
            running_time = end_time - start_time
            f.write(f"Time: {running_time}\n\n")
            f.close()
    