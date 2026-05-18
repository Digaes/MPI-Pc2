# Parallel Word Counting with MPI

---
# Group: 
Diana García Escorcia 

John Ferrer Ávila 

Kesly Rodríguez Salcedo 

Alexy Salcedo Tete 

## 1. Problem Description

The goal of this lab is to count how many times each word in `consulta.txt` appears across a corpus of text files (`file_*.txt`), and report the top 10 most frequent words. The problem is solved in three stages: a sequential baseline, a first MPI version using static file distribution, and a second MPI version that reduces load imbalance through dynamic scheduling.

---

## 2. Environment and Execution Instructions

### Generate the dataset
```bash
# Linux / Mac
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py

# Windows CMD
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 python /app/generator.py

# Windows PowerShell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/generator.py
```

### Check available cores in the container
```bash
docker run --rm augustosalazar/slim-mpi:2 nproc
```
> Result: **2 cores**

### Run the sequential baseline
```bash
# Windows CMD
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py

# Windows PowerShell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```

### Run MPI version 1 (static distribution)
```bash
# Windows CMD — replace N with 1, 2, 4, or 8
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np N python /app/mpi1.py

# Windows PowerShell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np N python /app/mpi1.py
```

### Run MPI version 2 (dynamic load balancing)
```bash
# Windows CMD — replace N with 1, 2, 4, or 8
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np N python /app/mpi2.py

# Windows PowerShell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np N python /app/mpi2.py
```

### Run all experiments automatically
```bash
# Windows CMD
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 sh /app/run_all.sh

# Windows PowerShell
docker run --rm -v "${PWD}:/app" augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

---

## 3. Experimental Plan

The experiment follows these steps:

1. Run the sequential baseline and record T_seq as the reference.
2. Run mpi1.py with p ∈ {1, 2, 4, 8}, performing 3 runs each and averaging the times.
3. Observe local processing times per rank to detect load imbalance.
4. Run mpi2.py with the same configurations to evaluate the dynamic scheduling improvement.
5. Compute Speedup (S_p = T_seq / T_p) and Efficiency (E_p = S_p / p) for both versions.
6. Compare both implementations and draw conclusions.

---

## 4. Sequential Baseline

The sequential baseline (`baseline_secuencial.py`) reads `consulta.txt` to build a set of target words, then iterates over all `file_*.txt` files in the dataset directory one by one, counting how many times each query word appears. All results are accumulated in a single global counter, and the top 10 most frequent words are printed at the end along with the total execution time. The results are saved to `baseline_results.csv` for comparison with the parallel versions.

---

## 5. MPI Version 1

The solution distributes the word-counting task across multiple parallel processes using the MPI standard from the mpi4py library. In the execution, process rank 0 orchestrates the work and all processes count locally the occurrences of the query words in their assigned files.

Only the root process reads the query file (`consulta.txt`) and builds a set of target words — this avoids redundant I/O from every process accessing the file system simultaneously. The query words are then broadcast to all processes in a single collective operation, and each process independently reconstructs the target word set for its local lookups.

Rank 0 scans the `dataset/` directory and builds a list of all `file_*.txt` paths, then broadcasts it to every process. Files are distributed statically using round-robin slicing (`all_files[rank::size]`), so each process independently reads its assigned files and counts query word occurrences line by line.

Finally, partial results are gathered in rank 0 using `comm.gather()`. Rank 0 merges all partial counters into a single global counter, calculates totals, identifies the 10 most frequent query words, saves results to a CSV file, and prints the top 10 including the total execution time.

---

## 6. Test Procedure

Each configuration was run 3 times and the average execution time was recorded. The sequential baseline was run once to obtain T_seq. Speedup and Efficiency were computed as:

- **S_p = T_seq / T_p**
- **E_p = S_p / p**

Local processing times per rank were recorded to evaluate load balance across processes.

---

## 7. Results

### Top 10 most frequent words

| Rank | Word | Count |
|------|------|-------|
| 1  | a         | 785774 |
| 2  | para      | 392156 |
| 3  | sus       | 228913 |
| 4  | otros     | 105530 |
| 5  | ante      |  99832 |
| 6  | unos      |  88794 |
| 7  | otra      |  83901 |
| 8  | vosotros  |  61617 |
| 9  | mios      |  58420 |
| 10 | tuya      |  56635 |

> ✅ The output above matches the sequential baseline exactly. Both `mpi1.py` and `mpi2.py` produce identical Top 10 results to `baseline_secuencial.py`.

---

## 8. Sequential Baseline Timing

| Execution time (s) |
|--------------------|
| 115.312768 |

---

## 9. MPI Version 1 Timing Results

**Container cores (`nproc`):** 2
**T_seq:** 115.312768 s

| p | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg T_p (s) | Speedup S_p | Efficiency E_p |
|---|-----------|-----------|-----------|-------------|-------------|----------------|
| 1 | 175.478973 | 68.343805 | 63.570830 | 102.464536 | 1.125 | 1.125 |
| 2 | 27.682803 | 24.875660 | 25.542109 | 26.033524 | 4.429 | 2.215 |
| 4 | 18.585941 | 19.706927 | 16.815953 | 18.369607 | 6.277 | 1.569 |
| 8 | 16.927521 | 16.233906 | 16.455516 | 16.538981 | 6.972 | 0.872 |

---

## 10. Load Imbalance Evidence

Running mpi1.py with p=4, the local processing times per rank across 3 runs were:

**Run 1:**
| Rank | Local processing time (s) |
|------|--------------------------|
| 0 | 11.086981 |
| 1 | 11.143025 |
| 2 | 11.529358 |
| 3 | 11.389847 |

**Run 2:**
| Rank | Local processing time (s) |
|------|--------------------------|
| 0 | 10.908818 |
| 1 | 10.924082 |
| 2 | 11.191672 |
| 3 | 11.156631 |

**Run 3:**
| Rank | Local processing time (s) |
|------|--------------------------|
| 0 | 9.459858 |
| 1 | 9.431577 |
| 2 | 9.802059 |
| 3 | 9.728296 |

The local processing times across ranks are very similar — differences of less than 0.5 seconds — confirming that the dataset files are uniform in size and the static round-robin distribution already achieves a naturally balanced workload. No significant load imbalance was observed in this experiment.

---

## 11. MPI Version 2

The second MPI implementation improves upon the static distribution of mpi1 by adopting a dynamic master-worker scheduling strategy. Instead of pre-assigning a fixed set of files to each process, rank 0 acts as a master that distributes work in chunks of 10 files at a time to idle workers, which eliminates the load imbalance caused by files of varying sizes.

As in mpi1, rank 0 reads `consulta.txt` and broadcasts the query words to all processes. However, rather than broadcasting the file list and letting each process slice its own portion, the master maintains a work queue of all file paths and seeds each worker with its first chunk using point-to-point communication with TAG_WORK.

Workers enter a loop where they receive a chunk of files, count the occurrences of query words, and immediately send the partial results back to the master with TAG_RESULT. The master collects each result, merges it into the global counter, and dispatches the next available chunk to the now-idle worker. When no files remain, the master sends a TAG_STOP signal to terminate each worker.

### MPI Version 2 Timing Results

| p | Run 1 (s) | Run 2 (s) | Run 3 (s) | Avg T_p (s) | Speedup S_p | Efficiency E_p |
|---|-----------|-----------|-----------|-------------|-------------|----------------|
| 1 | 70.809952 | 71.593263 | 73.147511 | 71.850242 | 1.6049 | 1.6049 |
| 2 | 51.456380 | 50.692756 | 49.072593 | 50.407243 | 2.2876 | 1.1438 |
| 4 | 24.119526 | 22.903534 | 23.192056 | 23.405039 | 4.9267 | 1.2317 |
| 8 | 20.612233 | 19.017965 | 18.477337 | 19.369178 | 5.9534 | 0.7442 |

### Comparison: mpi1 vs mpi2

| p | Tp mpi1 | Tp mpi2 | Sp mpi1 | Sp mpi2 | Ep mpi1 | Ep mpi2 |
|---|---------|---------|---------|---------|---------|---------|
| 1 | 102.464536 | 71.850242 | 1.125 | 1.6049 | 1.125 | 1.6049 |
| 2 | 26.033524 | 50.407243 | 4.429 | 2.2876 | 2.215 | 1.1438 |
| 4 | 18.369607 | 23.405039 | 6.277 | 4.9267 | 1.569 | 1.2317 |
| 8 | 16.538981 | 19.369178 | 6.972 | 5.9534 | 0.872 | 0.7442 |

When comparing both MPI implementations, the results reveal an interesting trade-off between static and dynamic scheduling strategies. In terms of total execution time, mpi1 outperforms mpi2 at p=2, p=4, and p=8, achieving 26.03s, 18.37s, and 16.54s respectively compared to mpi2's 50.41s, 23.41s, and 19.37s. However, at p=1 mpi2 is notably faster — 71.85s versus 102.46s — because with a single process mpi2 processes all files directly without any master-worker overhead. Regarding speedup, mpi1 consistently achieves higher values, peaking at 6.972 at p=8, while mpi2 peaks at 5.9534. Both implementations show sub-linear scaling beyond p=4, which is expected given the container only has 2 physical cores. In terms of efficiency, both implementations show superlinear values at lower process counts due to T_seq being larger than T_p, converging around 74-87% at p=8 as communication overhead grows. Regarding load balance, mpi2's chunk-based dynamic scheduling reduces the number of coordination messages compared to a file-by-file approach, yet mpi1 still outperforms it because the dataset files are uniform in size, meaning static round-robin distribution already achieves natural load balance without any coordination overhead.

---

## 12. Analysis

### Did the first MPI implementation improve execution time compared to the sequential baseline?

Yes, the first MPI implementation consistently reduced execution time compared to the sequential baseline. The sequential baseline recorded an execution time of 115.312768 seconds, while the MPI implementation achieved an average of 102.464536 seconds with 1 process, 26.033524 seconds with 2 processes, 18.369607 seconds with 4 processes, and 16.538981 seconds with 8 processes. The most significant improvement was observed going from 1 to 2 processes, where execution time dropped by 75%. These results confirm that parallelizing the workload across multiple processes with MPI produces a meaningful reduction in total execution time.

### Was the observed speedup linear?

The observed speedup was not linear, because it would imply S_p = p, meaning that 2 processes should have S_p = 2, 4 processes S_p = 4, and 8 processes S_p = 8. In practice, the speedups were 4.429, 6.277, and 6.972, respectively. While the gains are considerable with a smaller number of processes, the speedup stabilizes as p increases — going from p=4 to p=8 only improved the speedup from 6.277 to 6.972, despite doubling the number of processes. This behavior indicates that the sequential parts of the program, such as reading and transmitting the query file, gathering partial results, and merging the final counter in rank 0, impose an upper limit on the achievable speedup regardless of how many processes are added.

### Is there evidence of load imbalance? How was it observed?

Yes, there is evidence of load imbalance in the first MPI implementation, because the static distribution assigns files to processes without considering their actual size. Since the corpus files are not guaranteed to be the same size, some processes may receive a disproportionately larger volume of text to process than others. This imbalance manifests in the local processing times per-process printed during execution, so that if some ranks finish significantly earlier than others, they remain idle waiting at the `comm.gather()` barrier while the slowest process determines the total execution time. However, in this specific experiment the local processing times were very similar across ranks — differing by less than 0.5 seconds — because the dataset files are uniform in size. The efficiency drop between p=2 and p=8 still supports the presence of some overhead, partly due to communication costs growing with more processes.

### Did the second implementation reduce load imbalance?

Yes, mpi2 was designed to reduce load imbalance through dynamic chunk-based scheduling, where the master distributes 10 files at a time to idle workers instead of pre-assigning a fixed portion upfront. This ensures that faster workers receive more work automatically rather than sitting idle waiting for slower ones to finish. However, in practice the load imbalance reduction was not clearly reflected in the execution times because the dataset generated by `generator.py` produces files of uniform size, meaning mpi1's static round-robin distribution already achieves a naturally balanced workload. In a scenario with heterogeneous file sizes, mpi2's dynamic scheduling would show a more significant advantage.

### Did the improved distribution strategy produce a real performance improvement?

The improved distribution strategy produced a partial performance improvement. At p=1, mpi2 outperforms mpi1 significantly — 71.85s versus 102.46s — because with a single process mpi2 skips the master-worker coordination entirely and processes all files directly. However, for p=2, p=4, and p=8, mpi1 consistently achieves better execution times than mpi2. This suggests that the chunk-based dynamic scheduling, while reducing coordination overhead compared to a file-by-file approach, still introduces enough master-worker communication cost to outweigh the load balancing benefit on a uniform dataset. Therefore, the real performance improvement was only observed at p=1, while for higher process counts the static distribution of mpi1 proved more efficient under these specific experimental conditions.

### What limitations affected your experiment?

The main limitation of this experiment was the number of physical cores available in the container — with only 2 cores reported by `nproc`, running with p=4 and p=8 required oversubscription, meaning multiple processes competed for the same physical cores instead of running truly in parallel. This directly limits the achievable speedup and explains why performance gains stop growing beyond p=2. Additionally, significant execution time variability was observed across repeated runs — for example at p=1 in mpi1, times ranged from 175.47s down to 63.57s — which can be attributed to the Docker container sharing resources with the host operating system, meaning background system processes, memory pressure, and CPU scheduling decisions outside the container's control directly affected execution times between runs. Finally, since the dataset files are all generated with uniform size and content distribution, the workload was already naturally balanced across processes, which made it difficult to appreciate the real advantage of mpi2's dynamic scheduling strategy in practice.

---

## 13. Conclusions

Both MPI implementations successfully reduced execution time compared to the sequential baseline, with mpi1 achieving the most consistent improvements — dropping from 115.31s sequentially to 16.54s at p=8, a speedup of 6.972. The most important problem observed in the first parallel version was not load imbalance per se, but rather the limitation imposed by oversubscription — with only 2 physical cores available, adding more processes beyond p=2 produced diminishing returns as processes competed for the same hardware resources rather than running truly in parallel.

The second version addressed load imbalance through dynamic chunk-based scheduling, and while it did improve performance at p=1 compared to mpi1, it did not outperform the static distribution at higher process counts. This outcome is explained by the uniform nature of the dataset, where mpi1's round-robin distribution already achieves natural balance, and mpi2's coordination overhead becomes the dominant cost factor.

Based on the evidence collected, the parallel implementations clearly improved execution time over the sequential baseline, with the greatest gains at p=2 where the available physical cores were fully utilized. Beyond that, the hardware constraints of the experimental environment limited further scaling, highlighting that the effectiveness of a parallel solution depends not only on the algorithm design but also on the underlying hardware and workload characteristics.