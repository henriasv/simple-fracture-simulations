/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_biharmonic_smoothed.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairBiharmonicSmoothed::PairBiharmonicSmoothed(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairBiharmonicSmoothed::~PairBiharmonicSmoothed()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(k1);
    memory->destroy(k2);
    memory->destroy(r_0);
    memory->destroy(r_on);
    memory->destroy(r_c);
    memory->destroy(smoothing);
  }
}

/* ---------------------------------------------------------------------- */

void PairBiharmonicSmoothed::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq, forcebs,factor_lj;
  double r, r1, rexpfac, springfac;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        rexpfac = 1.0/(exp(r*smoothing[itype][jtype]/r_c[itype][jtype]-smoothing[itype][jtype])+1);
        
        
        if (r < r_on[itype][jtype]) {
          springfac = k1[itype][jtype]*(r-r_0[itype][jtype]);
        }
        else {
          r1 = r_on[itype][jtype]-k1[itype][jtype]/k2[itype][jtype]*(r_on[itype][jtype]-r_0[itype][jtype]);
          springfac = k2[itype][jtype]*(r-r1);
        }
        forcebs = springfac*rexpfac;
        fpair = -factor_lj*forcebs;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = 0;
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(k1,n+1,n+1,"pair:k1");
  memory->create(k2,n+1,n+1,"pair:k2");
  memory->create(r_0,n+1,n+1,"pair:r_0");
  memory->create(r_on,n+1,n+1,"pair:r_on");
  memory->create(r_c,n+1,n+1,"pair:r_c");
  memory->create(smoothing,n+1,n+1,"pair:smoothing");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::coeff(int narg, char **arg)
{
  if (narg != 8)
    error->all(FLERR,"Incorrect number of args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double k1_one = force->numeric(FLERR, arg[2]);
  double k2_one = force->numeric(FLERR, arg[3]);
  double r_0_one = force->numeric(FLERR, arg[4]);
  double r_on_one = force->numeric(FLERR, arg[5]);
  double r_c_one = force->numeric(FLERR, arg[6]);
  double smoothing_one = force->numeric(FLERR, arg[7]);


  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      k1[i][j] = k1_one; 
      k2[i][j] = k2_one; 
      r_0[i][j] = r_0_one;  
      r_on[i][j] = r_on_one;  
      r_c[i][j] = r_c_one;  
      smoothing[i][j] = smoothing_one;  
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBiharmonicSmoothed::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  k1[j][i] = k1[i][j]; 
  k2[j][i] = k2[i][j]; 
  r_0[j][i] = r_0[i][j];  
  r_on[j][i] = r_on[i][j];  
  r_c[j][i] = r_c[i][j];
  smoothing[j][i] = smoothing[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    etail_ij = 0; // Should be implemented
    ptail_ij = 0; // Should be implemented
  }

  return cut_global;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&k1[i][j],sizeof(double),1,fp);
        fwrite(&k2[i][j],sizeof(double),1,fp);
        fwrite(&r_0[i][j],sizeof(double),1,fp);
        fwrite(&r_on[i][j],sizeof(double),1,fp);
        fwrite(&r_c[i][j],sizeof(double),1,fp);
        fwrite(&smoothing[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&k1[i][j],sizeof(double),1,fp);
          fread(&k2[i][j],sizeof(double),1,fp);
          fread(&r_0[i][j],sizeof(double),1,fp);
          fread(&r_on[i][j],sizeof(double),1,fp);
          fread(&r_c[i][j],sizeof(double),1,fp);
          fread(&smoothing[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&k1[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&k2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r_0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r_on[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r_c[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&smoothing[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g\n",i,k1[i][i],k2[i][i],r_0[i][i],r_on[i][i], r_c[i][i], smoothing[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairBiharmonicSmoothed::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g\n",i,j,
              k1[i][j],k2[i][j],r_0[i][j],r_on[i][j],r_c[i][j], smoothing[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairBiharmonicSmoothed::single(int /*i*/, int /*j*/, int itype, int jtype,
                        double rsq, double /*factor_coul*/, double factor_lj,
                        double &fforce)
{
  
  return 0;
}

/* ---------------------------------------------------------------------- */

