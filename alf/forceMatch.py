#! /usr/bin/env python

import numpy as np 
import os
import shutil
from pathlib import Path
from scipy.sparse.linalg import cg, LinearOperator
import time

def read_analysis_dir(analysis_dir, ndupl, nreps, skip):
  L_traj = []
  dUdL_traj = []
  for trial in range(ndupl): 
    for rep in range(nreps): 
      lFile = analysis_dir.absolute() / f"data/Lambda.{trial}.{rep}.dat"
      lfFile = analysis_dir.absolute() / f"data/ForceFieldForce.{trial}.{rep}.dat"
      lData = np.loadtxt(lFile)[::skip]
      lfData = np.loadtxt(lfFile)[::skip]
      L_traj.append(lData)
      dUdL_traj.append(lfData)
  return L_traj, dUdL_traj

# Filter out data and fill array with active subs
def filter_data(L_traj, dUdL_traj, max_n_active):
  which_active = L_traj > 0
  n_actives = np.sum(which_active, axis=1)
  mask = n_actives != 1
  endstate_counts = np.sum(L_traj[~mask], axis=0)
  if max_n_active != -1:
    mask &= n_actives <= max_n_active
  return L_traj[mask], dUdL_traj[mask], endstate_counts

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
  # x -> xij*li*(1 - exp(-5.56*lj)) 
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

  return alpha0, basis_identity, basis_bounds


def forceMatch(alf_info, Ff, verbose=True):
  start_program = time.perf_counter()
  if not alf_info['impcons'] == 'fnpwise2026':
    print("Error: this script only works with fnpwise2026 constraint")
    quit(1)
  if not alf_info['bias'] == 'bcxs2018':
    print("Error: this script only works with bcxs2018 bias for now.")
    quit(1)
  nsubs = alf_info['nsubs'] # array of nsubs at each site 
  nblocks = alf_info['nblocks']
  nsite = nsubs.shape[0]
  if nsite != 1:
    print("Error: this script is not set up for multi-site calculations")
    quit(1)

  analysis_dir = Path('./') # runflat leaves you in current analysis directory
  nreps = alf_info['nreps']
  k0 = .01 # Fraction of nonlinear ALF's regularization
  skip = 1
  max_n_active = nblocks

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

  # Read files from analysis_dir/data/*
  start_load = time.perf_counter()
  L_traj, dUdL_traj = read_analysis_dir(analysis_dir, ndupl, nreps, skip)
  L_traj = np.vstack(L_traj)
  dUdL_traj = np.vstack(dUdL_traj)
  # Filter out unwanted data (end-states and n_active > max_n_active)
  L_traj, dUdL_traj, endstate_counts = filter_data(L_traj, dUdL_traj, max_n_active)
  end_load = time.perf_counter()
  
  # Build definitions of basis coefficent vector
  n_linear = int(nblocks)
  n_quad = int(nblocks*(nblocks-1)/2)
  per_path = 5 if alf_info['bias'] == 'bcxs2018' else 7
  n_basis = n_linear+per_path*n_quad
  alpha0, basis_identity, basis_bounds = build_alpha(alf_info, analysis_dir, nblocks, n_basis, n_quad)

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

  # Helper functions for processing samples
  K = 5.56
  if alf_info['bias']=='bcxs2018':
    ALPHA=0.017
  elif alf_info['bias']=='bcxstu2026':
    ALPHA=0.012
  I = basis_substituents[:, 0]
  J = basis_substituents[:, 1]
  P = n_basis
  masks = {k: basis_identity == k for k in ("b", "c", "xu", "xl", "su", "sl")}
  def basis_grads(L):   # L: (S, nblocks) -> gi, gj: (S, P)
      li, lj = L[:, I], L[:, J] # (S, P) filled with lambda pairs for each basis function
      gi, gj = np.zeros_like(li), np.zeros_like(li)
      m = masks["b"];  gi[:, m] = 1
      m = masks["c"];  gi[:, m] = lj[:, m];  gj[:, m] = li[:, m]
      m = masks["xu"]; e = np.exp(-K*li[:, m]); gi[:, m] = K*lj[:, m]*e; gj[:, m] = 1 - e
      m = masks["xl"]; e = np.exp(-K*lj[:, m]); gi[:, m] = 1 - e;        gj[:, m] = K*li[:, m]*e
      m = masks["su"]; q = li[:, m] + ALPHA; gi[:, m] = ALPHA*lj[:, m]/q**2; gj[:, m] = li[:, m]/q
      m = masks["sl"]; q = lj[:, m] + ALPHA; gi[:, m] = lj[:, m]/q;        gj[:, m] = ALPHA*li[:, m]/q**2
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

  # Accumulate samples from this run into Amat.dat and bvec.dat
  start_Ab_read = time.perf_counter()
  path_Amat = Path(f"../analysis{Ff-1}/Amat.dat")
  path_bvec = Path(f"../analysis{Ff-1}/bvec.dat")
  path_accum = Path(f"../analysis{Ff-1}/accumulated_count.dat")
  if path_Amat.is_file():
    A = np.loadtxt(path_Amat, ndmin=2)
    assert A.shape == (P, P), f"{path_Amat} has shape {A.shape}, expected {(P, P)}"
  else:
    A = np.zeros((P, P))
  if path_bvec.is_file():
    bvec = np.loadtxt(path_bvec, ndmin=1)
    assert bvec.shape == (P,), f"{path_bvec} has shape {bvec.shape}, expected {(P,)}"
  else:
    bvec = np.zeros(P)
  if path_accum.is_file():
    accumulated_count = np.loadtxt(path_accum, ndmin=1, dtype=np.int64)
    assert accumulated_count.shape == (1,), f"{path_accum} has shape {accumulated_count.shape}, expected {(1,)}"
  else:
    accumulated_count = np.zeros(1, dtype=np.int64)
  end_Ab_read = time.perf_counter()
 
  # Accumulation into A and b by sample chunks
  chunk = 1024
  start_accumulate = time.perf_counter()
  for lo in range(0, L_traj.shape[0], chunk):
    sl = slice(lo, lo + chunk)
    accumulate(L_traj[sl], dUdL_traj[sl], A, bvec)
  accumulated_count += L_traj.shape[0]
  end_accumulate = time.perf_counter()
  # Save to current analysis directory
  start_Ab_write = time.perf_counter()
  path_Amat = Path(f"Amat.dat")
  path_bvec = Path(f"bvec.dat")
  path_accum = Path(f"accumulated_count.dat")
  np.savetxt(path_Amat, A, fmt="%.17e")
  np.savetxt(path_bvec, bvec, fmt="%.17e")
  np.savetxt(path_accum, accumulated_count, fmt="%d")
  end_Ab_write = time.perf_counter()
  # TODO: Remove A.dat and b.dat from ../analysis{Ff-1}/ to save space for larger A files

  # Per-basis-type regularization strengths 
  k0_by_type = {
    "b":  k0/4, "c":  k0/64, "xu": k0/4, 
    "xl": k0/4,"su": k0, "sl": k0}
  k0_vec = np.array([k0_by_type[t] for t in basis_identity])   # shape (n_basis,)
  # Scale k0_vec to have same weight as all samples (i.e. place regularization inside sample sum)
  k0_vec *= accumulated_count
 
  # (A + diag(k0)) alpha = bvec + k0 * alpha0 
  start_solver = time.perf_counter()
  Areg = A + np.diag(k0_vec)
  rhs = bvec + k0_vec * alpha0
  dinv = 1.0 / np.diag(Areg) # Preconditioner
  M = LinearOperator(Areg.shape, matvec=lambda v: dinv * v)
  # Track iterations
  iters = {"n": 0}
  def count(_xk):
      iters["n"] += 1
  # Solve (Amat@alpha = bvec) with CG
  soln_vector, info = cg(Areg, rhs, x0=alpha0, rtol=1e-10, maxiter=10 * n_basis,
                         M=M, callback=count)
  if info != 0:
      print(f"WARNING: CG did not converge (info={info})")
  rel_res = np.linalg.norm(Areg @ soln_vector - rhs) / np.linalg.norm(rhs)
  print(f"CG iterations: {iters['n']}, relative residual: {rel_res:.3e}")
  end_solver = time.perf_counter()

  # Subtract reference linear parameter
  soln_vector[:nblocks] -= soln_vector[0]
  
  # shift linear coefficent if a substituent hasn't been sampled in the last 10 cycles
  path_since = Path("cycles_since_sampled.dat")
  prev_since = Path(f"../analysis{Ff-1}/cycles_since_sampled.dat")
  since = np.loadtxt(prev_since, ndmin=1) if prev_since.is_file() else np.zeros(nblocks)
  n_min = 10 # Threshold for "being sampled"
  sampled = endstate_counts >= n_min
  since = np.where(sampled, 0, since + 1)
  np.savetxt(path_since, since.reshape(1, -1), fmt="%d")
  dG = np.where(since >= 5 , .1, 0.0)
  dG = np.where(since >= 10, .5, dG)
  dG -= dG[0]

  # Write differences between old and new alphas: b.dat, c.dat, ...
  alpha_new = soln_vector.copy()
  alpha_new -= alpha0
  alpha_new[:nblocks] += dG # Soln_vector is a model of the FES, so add

  update_b = alpha_new[masks['b']]
  update_b = update_b.reshape(1, -1)
  np.savetxt(analysis_dir.absolute() / 'b.dat', update_b, fmt="%7.2f")
  np.savetxt(analysis_dir.absolute() / 'b_hist.dat', dG.reshape(1, -1), fmt="%7.2f")
  update_c = np.zeros((nblocks,nblocks))
  update_c[np.triu_indices_from(update_c, k=1)] = alpha_new[masks['c']]
  np.savetxt(analysis_dir.absolute() / 'c.dat', update_c, fmt="%7.2f")
  update_x = np.zeros((nblocks, nblocks))
  update_x[np.triu_indices_from(update_x, k=1)] = alpha_new[masks['xu']]
  update_x[np.tril_indices_from(update_x, k=-1)] = alpha_new[masks['xl']]
  np.savetxt(analysis_dir.absolute() / 'x.dat', update_x, fmt="%7.2f")
  update_s = np.zeros((nblocks, nblocks))
  update_s[np.triu_indices_from(update_s, k=1)] = alpha_new[masks['su']]
  update_s[np.tril_indices_from(update_s, k=-1)] = alpha_new[masks['sl']]
  np.savetxt(analysis_dir.absolute() / 's.dat', update_s, fmt="%7.2f")
  end_program = time.perf_counter()

  # Print timings
  print(f"Load lambda and lambda force trajectories: {end_load - start_load:10.2f}s")
  print(f"Read A and b: {end_Ab_read - start_Ab_read:10.2f}s")
  print(f"Accumulate into A and b: {end_accumulate -  start_accumulate:10.2f}s")
  print(f"Write A and b: {end_Ab_write - start_Ab_write:10.2f}s")
  print(f"Solve linear system: {end_solver - start_solver:10.2f}s")
  print(f"Total program time: {end_program - start_program:10.2f}s")
