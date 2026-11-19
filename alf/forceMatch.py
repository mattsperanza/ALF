#! /usr/bin/env python

import numpy as np 
import os
from pathlib import Path
from scipy.sparse.linalg import cg, LinearOperator
import time

# Constant defined globally so it can be accessed throughout ALF
NBINS_FM_HIST = 128

def check_options_settings(alf_info):
  if not alf_info['impcons'] == 'fnpwise2026':
    print("Error: force matching script only works with fnpwise2026 constraint")
    quit(1)
  if not alf_info['bias'] == 'bcxs2018':
    print("Error: this script only works with bcxs2018 bias for now.")
    quit(1)
  if alf_info['nsubs'].shape[0] != 1:
    print("Error: this script is not set up for multi-site calculations")
    quit(1)

def determine_ndupl(Ff):
  # Determine ndupl
  if os.path.isdir('../run'+str(Ff)):
    ndupl=1
  else:
    alphabet='abcdefghijklmnopqrstuvwxyz'
    ndupl=0
    for i in range(0,len(alphabet)):
      if os.path.isdir('../run'+str(Ff)+alphabet[i]):
        ndupl+=1
    if ndupl==0:
      print("Error, not flattening or production")
      quit(1)
  return ndupl

def build_alpha(alf_info, analysis_dir, nblocks, n_basis, n_quad):
  b = np.loadtxt(analysis_dir.absolute() / "b_prev.dat")
  c = np.loadtxt(analysis_dir.absolute() / "c_prev.dat")
  x = np.loadtxt(analysis_dir.absolute() / "x_prev.dat")
  s = np.loadtxt(analysis_dir.absolute() / "s_prev.dat")
  if alf_info['bias'] == 'bcxstu2026':
    t = np.loadtxt(analysis_dir.absolute() / "t_prev.dat")
    u = np.loadtxt(analysis_dir.absolute() / "u_prev.dat")
  # build alpha vector definition
  alpha0 = np.zeros(n_basis)
  bound = 0
  basis_bounds = [0]
  basis_identity = np.full((n_basis,), "", dtype="<U3")
  # b -> bi*li
  alpha0[0:nblocks]=b
  basis_identity[0:nblocks]="b"
  bound = nblocks
  basis_bounds.append(bound)
  # c -> cij*li*lj
  alpha0[bound:bound+n_quad]=c[np.triu_indices(nblocks, k=1)]
  basis_identity[bound:bound+n_quad]="c"
  bound = bound+n_quad
  basis_bounds.append(bound)
  # x -> xij*(1 - exp(-5.56*li))*lj
  alpha0[bound:bound+n_quad]=x[np.triu_indices(nblocks, k=1)]
  basis_identity[bound:bound+n_quad]="xu"
  bound = bound+n_quad
  alpha0[bound:bound+n_quad]=x[np.tril_indices(nblocks, k=-1)]
  basis_identity[bound:bound+n_quad]="xl"
  bound = bound+n_quad
  basis_bounds.append(bound)
  # s -> sij*li*lj/(li + alpha)
  alpha0[bound:bound+n_quad]=s[np.triu_indices(nblocks, k=1)]
  basis_identity[bound:bound+n_quad]="su"
  bound = bound+n_quad
  alpha0[bound:bound+n_quad]=s[np.tril_indices(nblocks, k=-1)]
  basis_identity[bound:bound+n_quad]="sl"
  bound = bound+n_quad
  basis_bounds.append(bound)
  # t -> tij*li*lj/(li - (1+alpha))
  # u -> tij*li*lj^2/(li + alpha)

  basis_masks = {k: basis_identity == k for k in ("b", "c", "xu", "xl", "su", "sl")}

  return alpha0, basis_identity, basis_bounds, basis_masks

# Create maps from block->basis and basis->block
def map_block_to_alpha(nblocks, per_path, n_basis, basis_bounds):
  # List of basis functions each substituent is a part of, and vice versa
  quad_indices = np.zeros((nblocks,nblocks), dtype=np.int32)
  quad_indices[np.triu_indices(nblocks, k=1)]=np.arange(0, int(nblocks*(nblocks-1)/2), dtype=np.int32)
  quad_indices[np.tril_indices(nblocks, k=-1)]=np.arange(int(nblocks*(nblocks-1)/2),int(2*nblocks*(nblocks-1)/2), dtype=np.int32)
  connected_count = 1 + per_path*(nblocks-1) # Each sub is part of this many basis functions
  connected_indices = np.zeros((nblocks, connected_count), dtype=np.int32)
  basis_substituents = np.zeros((n_basis, 2), dtype=np.int32)
  counts = np.zeros((n_basis,), dtype=np.int32)
  for i in range(nblocks):
    connected_indices[i][0] = i  # b
    tmp = np.concatenate([quad_indices[i, i+1:].ravel(), quad_indices[:i, i].ravel()]) 
    connected_indices[i][1:nblocks] = basis_bounds[1] + tmp  # c - only upper
    tmp = np.concatenate([tmp, quad_indices[i, :i].ravel(), quad_indices[i+1:, i].ravel()])
    connected_indices[i][nblocks:3*nblocks-2] = basis_bounds[2] + tmp  # x
    connected_indices[i][3*nblocks-2:5*nblocks-4] = basis_bounds[3] + tmp  # s
    for p in connected_indices[i]:
      basis_substituents[p, counts[p]] = i
      counts[p] += 1
  # Check that each basis function has 1 or 2 substituents, and that all basis functions have been counted
  assert (counts[:nblocks] == 1).all() and (counts[nblocks:] == 2).all()
  lin_terms = counts == 1
  basis_substituents[lin_terms, 1] = basis_substituents[lin_terms, 0]
  return basis_substituents, connected_indices

# File reading for force matching
def read_FM_data(analysis_dir, ndupl, nreps, skipFM):
  L_traj = []
  dUdL_traj = []
  for trial in range(ndupl): 
    for rep in range(nreps): 
      lFile = analysis_dir.absolute() / f"data/Lambda.{trial}.{rep}.dat"
      lfFile = analysis_dir.absolute() / f"data/ForceFieldForce.{trial}.{rep}.dat"
      lData = np.loadtxt(lFile)[::skipFM] 
      lfData = np.loadtxt(lfFile)[::skipFM]
      L_traj.append(lData)
      dUdL_traj.append(lfData)
  return L_traj, dUdL_traj

# Filter out data and fill array with active subs
def filter_FM_data(L_traj, dUdL_traj, max_n_active):
  which_active = L_traj > 0
  n_actives = np.sum(which_active, axis=1)
  mask = n_actives != 1
  endstate_counts = np.sum(L_traj[~mask], axis=0)
  if max_n_active != -1:
    mask &= n_actives <= max_n_active
  return L_traj[mask], dUdL_traj[mask], endstate_counts, n_actives

def accumulate_FM_data(alf_info, A, bvec, L_traj, dUdL_traj, nblocks, n_basis, basis_substituents, basis_masks):
  # Helper functions for processing samples
  K = 5.56
  if alf_info['bias']=='bcxs2018':
    ALPHA=0.017
  elif alf_info['bias']=='bcxstu2026':
    ALPHA=0.012
  I = basis_substituents[:, 0]
  J = basis_substituents[:, 1]
  P = n_basis
  def basis_grads(L):   # L: (S, nblocks) -> gi, gj: (S, P)
      li, lj = L[:, I], L[:, J] # (S, P) filled with lambda pairs for each basis function
      gi, gj = np.zeros_like(li), np.zeros_like(li)
      m = basis_masks["b"];  gi[:, m] = 1
      m = basis_masks["c"];  gi[:, m] = lj[:, m];  gj[:, m] = li[:, m]
      m = basis_masks["xu"]; e = np.exp(-K*li[:, m]); gi[:, m] = K*lj[:, m]*e; gj[:, m] = 1 - e
      m = basis_masks["xl"]; e = np.exp(-K*lj[:, m]); gi[:, m] = 1 - e;        gj[:, m] = K*li[:, m]*e
      m = basis_masks["su"]; q = li[:, m] + ALPHA; gi[:, m] = ALPHA*lj[:, m]/q**2; gj[:, m] = li[:, m]/q
      m = basis_masks["sl"]; q = lj[:, m] + ALPHA; gi[:, m] = lj[:, m]/q;        gj[:, m] = ALPHA*li[:, m]/q**2
      # Partial derivatives of every sample and every basis function w.r.t. its component lambdas
      return gi, gj
  def accumulate(L, dUdL, A, bvec):
      d = 3 * L**(2/3)
      D = d.sum(1, keepdims=True)
      # dot = sum_i (dU/dLi * dLi/dTi) / sum_i (dLi/dTi)
      dot = (dUdL * d).sum(1, keepdims=True) / D
      # dU/dT = dL/dT * (dU/dL + dot)
      dUdT = d * (dUdL - dot)
  
      gi, gj = basis_grads(L)
      # sum_i (dfp/dLi * dLi/dTi) / sum_i (dLi/dTi) -> dfp/dLi is non-zero for only fp's lambdas
      gbar = (d[:, I]*gi + d[:, J]*gj) / D   # (S, P) (gj = 0 for linear terms)
  
      S = L.shape[0] # Chunk size
      G = np.zeros((S, P, nblocks)) # Final dimension only has 2 non-zero elements per (S, P)
      p_indices = np.arange(P)
      G[:, p_indices, I] = gi # G[:, p, I[p]] = g[:, p] for each p
      G[:, p_indices, J] += gj # only overlaps for linear terms, gj = 0 there
      # Elements of df_p/dTi|_s
      H = d[:, None, :] * (G - gbar[:, :, None])    # (S, P, nblocks)

      # b[p] += sum_s sum_i H[s,p,i]*dUdT[s,i]
      bvec += np.einsum("spi,si->p", H, dUdT)
      Hm = H.transpose(1, 0, 2).reshape(P, -1)      # (S, P, nblocks) -> (P, S, nblocks) -> (P, S*nblocks)
      A += Hm @ Hm.T # (P, P) A[p,l] += sum_s sum_i df_p/dTi|s * df_l/dTi|s
  chunk = 1024 # Chunk size
  for lo in range(0, L_traj.shape[0], chunk):
    sl = slice(lo, lo + chunk)
    accumulate(L_traj[sl], dUdL_traj[sl], A, bvec)


def forceMatch(alf_info, Ff, fpout, fperr, verbose=True):
  start_program = time.perf_counter()
  check_options_settings(alf_info)
  ndupl = determine_ndupl(Ff)
  analysis_dir = Path('./') # runflat leaves you in current analysis directory
  # Load MSLD parameters
  nsubs = alf_info['nsubs'] 
  nblocks = alf_info['nblocks']
  nreps = alf_info['nreps']
  kT=0.001987*alf_info['temp']

  ###################
  ## Parameters  ####
  ###################
  skipFM = 1                  # Frame skipping (skipping potentially saves lots of time without sacrificing fit)
  max_n_active = nblocks      # Filter out samples with more than this many n_active
  n_min = 10                  # Threshold for "being sampled"
  d = .95                     # Exponential Decay Base
  k_forceMatch = .8           # Force sample weight
  k_hist = .2                 # Endstate updates
  k0 = 0.01                   # Fraction of nonlinear ALF's regularization
  k0_by_type = {"b":  k0/4, "c":  k0/64, "xu": k0/4, "xl": k0/4,"su": k0, "sl": k0}

  # Build definitions of basis coefficent vector
  n_linear = int(nblocks); n_quad = int(nblocks*(nblocks-1)/2)
  per_path = 5 if alf_info['bias'] == 'bcxs2018' else 7
  n_basis = n_linear+per_path*n_quad
  alpha0, basis_identity, basis_bounds, basis_masks = build_alpha(alf_info, analysis_dir, nblocks, n_basis, n_quad)
  basis_substituents, connected_indices = map_block_to_alpha(nblocks, per_path, n_basis, basis_bounds)

  ###################
  ## Data Matrix  ###
  ###################
  start_Ab_read = time.perf_counter()
  path_Amat = Path(f"../analysis{Ff-1}/Amat.dat")
  path_bvec = Path(f"../analysis{Ff-1}/bvec.dat")
  path_accum = Path(f"../analysis{Ff-1}/accumulated_count.dat")
  # All or none file check
  present = [path_Amat.is_file(), path_bvec.is_file(), path_accum.is_file()]
  assert all(present) or not any(present), (f"Inconsistent accumulated files in analysis{Ff-1}: "
                          f"Amat={present[0]}, bvec={present[1]}, accumulated_count={present[2]}")
  A = np.loadtxt(path_Amat, ndmin=2) if path_Amat.is_file() else np.zeros((n_basis, n_basis))
  bvec = np.loadtxt(path_bvec, ndmin=1) if path_bvec.is_file() else np.zeros(n_basis)
  accumulated_count = np.loadtxt(path_accum, ndmin=1, dtype=np.float64) if path_accum.is_file() else np.zeros(1)
  assert A.shape == (n_basis, n_basis), f"{path_Amat} has shape {A.shape}, expected {(n_basis, n_basis)}"
  assert bvec.shape == (n_basis,), f"{path_bvec} has shape {bvec.shape}, expected {(n_basis,)}"
  assert accumulated_count.shape == (1,), f"{path_accum} has shape {accumulated_count.shape}, expected {(1,)}"
  end_Ab_read = time.perf_counter()
  # Exponental sample weight decay (before accumulating)
  A *= d; bvec *= d; accumulated_count *= d

  ###################
  ## Force Match  ###
  ###################
  # Read traj/force files from analysis_dir/data/*
  start_load = time.perf_counter()
  L_traj, dUdL_traj = read_FM_data(analysis_dir, ndupl, nreps, skipFM)
  L_traj = np.vstack(L_traj); dUdL_traj = np.vstack(dUdL_traj)
  # Filter out unwanted data for force matching (end-states and n_active > max_n_active)
  L_traj, dUdL_traj, endstate_counts, n_actives = filter_FM_data(L_traj, dUdL_traj, max_n_active)
  hist = np.bincount(n_actives, minlength=nblocks).astype(np.int32)
  print(f"Endstate Counts: {endstate_counts}", file=fpout)
  print(f"Counts in each n_active state: {hist}", file=fpout)
  print(f"Filtered FM L_traj shape: {L_traj.shape}, max_n_active={max_n_active}", file=fpout)
  end_load = time.perf_counter()
   # Keep track of how many iterations since each state was sampled
  prev_since = Path(f"../analysis{Ff-1}/cycles_since_sampled.dat")
  since = np.loadtxt(prev_since, ndmin=1) if prev_since.is_file() else np.zeros(nblocks)
  sampled = endstate_counts >= n_min
  since = np.where(sampled, 0, since + 1)
  np.savetxt("cycles_since_sampled.dat", since.reshape(1, -1), fmt="%d")
  # Keep track of accumulated endstate samples
  prev_endstate = Path(f"../analysis{Ff-1}/accumulated_endstates.dat")
  accum_endstate = np.loadtxt(prev_endstate, ndmin=1) if prev_endstate.is_file() else np.zeros(nblocks)
  accum_endstate = d*accum_endstate + endstate_counts
  np.savetxt("accumulated_endstates.dat", accum_endstate.reshape(1, -1), fmt="%.6f")

  # Accumulate samples into A and b by sample chunks
  start_accumulate = time.perf_counter()
  accumulate_FM_data(alf_info, A, bvec, L_traj, dUdL_traj, nblocks, n_basis, basis_substituents, basis_masks) 
  accumulated_count += L_traj.shape[0]
  end_accumulate = time.perf_counter()
  # Clean up after force matching to avoid confusion w/ densities
  del L_traj; del dUdL_traj; del endstate_counts; del n_actives
  # Save force matching system to current analysis directory
  start_Ab_write = time.perf_counter()
  np.savetxt("Amat.dat", A, fmt="%.17e")
  np.savetxt("bvec.dat", bvec, fmt="%.17e")
  np.savetxt("accumulated_count.dat", accumulated_count, fmt="%.17e")
  # Scale history (after saving) so all force samples have a fixed weight reguardless of the number of samples
  scale = k_forceMatch / max(accumulated_count[0], 1.0)
  A *= scale; bvec *= scale;
  # Remove A and b from analysis{Ff-2} to save space for larger A files
  if os.path.exists(f"../analysis{Ff-2}/Amat.dat"):
    os.remove(f"../analysis{Ff-2}/Amat.dat")
  if os.path.exists(f"../analysis{Ff-2}/bvec.dat"):
    os.remove(f"../analysis{Ff-2}/bvec.dat")
  end_Ab_write = time.perf_counter()

  ###################
  ## Histogram  #####
  ###################
  start_histogram_loss = time.perf_counter()
  # Read density traj from Lambda/Lambda.dat and un-normalized target weights from Lambda/weights.dat
  L_traj = np.loadtxt("Lambda/Lambda.dat", ndmin=2)
  weights = np.loadtxt("Lambda/weight.dat", ndmin=1)
  if L_traj.shape[0] != weights.shape[0]:
    print(f"Error: Lambda/Lambda.dat and Lambda/weight.dat do not have the same number of rows! L_traj: {L_traj.shape}, weights: {weights.shape}")
    quit(1)
  weights /= np.sum(weights) # Weights need to be normalized
  actives = L_traj > 0
  n_actives = np.sum(actives, axis=1)
  endstate_mask = n_actives == 1
  endstate_counts = np.sum(L_traj[endstate_mask], axis=0)
  # sum_t w_{Kt} I_{i}
  endstate_weights = np.sum(actives[endstate_mask]*weights[endstate_mask, None], axis=0)
  print(f"Histogram FE Endstate Counts: {endstate_counts}", file=fpout)
  print(f"Histogram FE Endstate weights: {endstate_weights}", file=fpout)
  # Histogram FE vs. linear b cefficient loss
  actives_mask = (endstate_counts >= n_min) & (endstate_weights > 0)
  idx = np.flatnonzero(actives_mask)
  num_actives = len(idx)
  if num_actives > 1:
    # Histogram k_hist*(a0 + F - a + G)^2
    F = np.where(actives_mask, -kT*np.log(np.where(actives_mask, endstate_weights, 1)), 0)
    print(f"Histogram excess Fi: {np.where(actives_mask, F, np.nan)}", file=fpout)
    F += np.where(actives_mask, alpha0[:nblocks], 0)
    F = np.where(actives_mask, F - F[idx].mean(), 0)
    A_hist = np.zeros((nblocks, nblocks))
    A_hist[np.ix_(idx, idx)] = -k_hist/num_actives
    A_hist[idx, idx] += k_hist
    b_hist = k_hist*F 

    A[0:nblocks, 0:nblocks] += A_hist
    bvec[0:nblocks] += b_hist
  # TODO: Direct path histogram FE loss with MBAR weights? Is this required?
  end_histogram_loss = time.perf_counter()

  ###################
  ## Solver  ########
  ###################
  # Per-basis-type regularization
  # (A + diag(k0)) alpha = bvec + k0 * alpha0 
  k0_vec = np.array([k0_by_type[t] for t in basis_identity])   # shape (n_basis,)
  A = A + np.diag(k0_vec)
  bvec = bvec + k0_vec*alpha0
  # Solve (Amat@alpha = bvec) with CG
  start_solver = time.perf_counter()
  dinv = 1.0 / np.diag(A) # Diagonal preconditioner
  M = LinearOperator(A.shape, matvec=lambda v: dinv * v)
  iters = {"n": 0}
  def count(_xk):
    iters["n"] += 1
  soln_vector, info = cg(A, bvec, x0=alpha0, rtol=1e-10, maxiter=10 * n_basis,
                         M=M, callback=count)
  if info != 0:
    print(f"WARNING: CG did not converge (info={info})", file=fpout)
  rel_res = np.linalg.norm(A @ soln_vector - bvec) / np.linalg.norm(bvec)
  print(f"CG iterations: {iters['n']}, relative residual: {rel_res:.3e}", file=fpout)
  # Subtract reference linear parameter
  soln_vector[:nblocks] -= soln_vector[0]
  end_solver = time.perf_counter()

  # Write differences between old and new alphas: b.dat, c.dat, ...
  alpha_new = soln_vector.copy()
  alpha_new -= alpha0
  update_b = alpha_new[basis_masks['b']]
  update_b = update_b.reshape(1, -1)
  np.savetxt(analysis_dir.absolute() / 'b.dat', update_b, fmt="%7.2f")
  update_c = np.zeros((nblocks,nblocks))
  update_c[np.triu_indices_from(update_c, k=1)] = alpha_new[basis_masks['c']]
  np.savetxt(analysis_dir.absolute() / 'c.dat', update_c, fmt="%7.2f")
  update_x = np.zeros((nblocks, nblocks))
  update_x[np.triu_indices_from(update_x, k=1)] = alpha_new[basis_masks['xu']]
  update_x[np.tril_indices_from(update_x, k=-1)] = alpha_new[basis_masks['xl']]
  np.savetxt(analysis_dir.absolute() / 'x.dat', update_x, fmt="%7.2f")
  update_s = np.zeros((nblocks, nblocks))
  update_s[np.triu_indices_from(update_s, k=1)] = alpha_new[basis_masks['su']]
  update_s[np.tril_indices_from(update_s, k=-1)] = alpha_new[basis_masks['sl']]
  np.savetxt(analysis_dir.absolute() / 's.dat', update_s, fmt="%7.2f")
  end_program = time.perf_counter()

  # Print timings
  print("Force Matching Timings:", file=fpout)
  print(f"{end_load - start_load:7.1f}s    loading lambda and lambda force trajectories", file=fpout) # 4 spaces after 's'
  print(f"{end_Ab_read - start_Ab_read:7.1f}s    reading A and b", file=fpout)
  print(f"{end_accumulate -  start_accumulate:7.1f}s    accumulating into A and b", file=fpout)
  print(f"{end_Ab_write - start_Ab_write:7.1f}s    writing A and b", file=fpout)
  print(f"{end_histogram_loss - start_histogram_loss:7.1f}s    performing weighted histogram loss", file=fpout)
  print(f"{end_solver - start_solver:7.1f}s    solving linear system", file=fpout)
  print(f"{end_program - start_program:7.1f}s    total program time", file=fpout)

  print(f"Total force matching program time: {end_program - start_program:10.1f}s")
