#!/usr/bin/env python
# -*- coding: utf-8 -*-"
"""
Class handling the compilation of HypoDD.

    * "working_dir"/bin/hypoDD
    * "working_dir"/bin/ph2dt
    * "working_dir"/bin/hypoDD.inc

hypoDD and ph2dt are the binaries for the respective programs and hypoDD.inc is
the hypoDD.inc file used for the compilation.

If all three files are present and the hypoDD.inc that would be used for a new
compilation is identical to the one already present nothing will happen as the
end result would be the same.
"""
# import md5
import hashlib
import os
import shutil
import subprocess
import tarfile


# Specify the HypoDD version to be compiled.
HYPODD_ARCHIVE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "src", "HYPODD_2.1b.tar.gz")
)

# Note this Hash is from the tar.gz file you got, either change this to your
# correct one using the function call md5.md5(open_file.read()).hexdigest()
# or simply comment out the line: if md5_hash != HYPODD_MD5_HASH:
# WCC added the hash pour his version and changed test from "!=" to "not in"
HYPODD_MD5_HASHES = [
    "ac7fb5829abef23aa91f1f8a115e2b45",
    "94228305b2370c4f3371fc6cb76f92c5",
]
HYPODD_DIAGNOSTIC_PATCH_VERSION = "ddres-diagnostics-v3"


class HypoDDCompilationError(Exception):
    """
    Exception that will be raised if anything during the compilation does not
    occur as planned.
    """

    pass


class HypoDDCompiler(object):
    """
    Class handling the HypoDD compilation.

    Usage
    =====

    >>> hyp_comp = HypoDDCompiler("temp_dir")
    >>> hyp_comp.configure()
    >>> hyp_comp.make()
    """

    def __init__(self, working_dir, log_function):
        """
        :param working_dir: The working directory. Everything will happen in
            there.
        :param log_function: Function to use to log activity.
        """
        # Set the log function.
        self.log = log_function
        # Set the working dir and create it if necessary.
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        # Make sure the given HypoDD archive is valid.
        self.verify_archive()
        # Setup and determine all the necessary paths.
        self.determine_paths()
        self.is_configured = False

    def verify_archive(self):
        """
        Method that checks if the HypoDD archive exists and that its md5 has is
        valid.
        """
        if not os.path.exists(HYPODD_ARCHIVE):
            msg = "HypoDD archive file could not be found"
            raise HypoDDCompilationError(msg)
        # Check if the file is correct.
        with open(HYPODD_ARCHIVE, "rb") as open_file:
            md5_hash = hashlib.md5(open_file.read()).hexdigest()
        # if md5_hash not in HYPODD_MD5_HASHES:
        #     msg = "md5 hash of the HypoDD archive is not correct"
        #     raise HypoDDCompilationError(msg)

    def determine_paths(self):
        self.paths = {}
        # Binary dir.
        self.paths["binary_dir"] = os.path.join(self.working_dir, "bin")
        if not os.path.exists(self.paths["binary_dir"]):
            os.makedirs(self.paths["binary_dir"])
        # Output files.
        self.paths["hypoDD_binary"] = os.path.join(
            self.paths["binary_dir"], "hypoDD"
        )
        self.paths["ph2dt_binary"] = os.path.join(
            self.paths["binary_dir"], "ph2dt"
        )
        # The hypoDD.inc files produced by any potential previous runs. After
        # the run, the currently used hypoDD.inc file will be copied there.
        self.paths["old hypoDD.inc file"] = os.path.join(
            self.paths["binary_dir"], "hypoDD.inc"
        )
        self.paths["diagnostic_patch_stamp"] = os.path.join(
            self.paths["binary_dir"], "hypoDD_diagnostic_patch.txt"
        )
        # Where to unpack the archive.
        self.paths["hypodd_unpack_dir"] = os.path.join(
            self.working_dir, "hypodd_src"
        )
        self._set_source_paths(
            os.path.join(self.paths["hypodd_unpack_dir"], "HYPODD")
        )

    def _set_source_paths(self, source_root):
        """
        Set paths inside the unpacked HypoDD source tree.

        Different HypoDD archives unpack to differently-cased top-level
        directories, e.g. HYPODD or HypoDD. The compiler should not care.
        """
        self.paths["source_root"] = source_root
        # Some paths in the unpacked archive.
        self.paths["make_directory"] = os.path.join(source_root, "src")
        # The resulting binaries directly after the compilation.
        self.paths["compiled_hypodd_binary"] = os.path.join(
            self.paths["make_directory"], "hypoDD", "hypoDD"
        )
        self.paths["compiled_ph2dt_binary"] = os.path.join(
            self.paths["make_directory"], "ph2dt", "ph2dt"
        )
        # The include directory.
        self.paths["include_dir"] = os.path.join(source_root, "include")
        # The hypoDD.inc file
        self.paths["hypoDD.inc"] = os.path.join(
            self.paths["include_dir"], "hypoDD.inc"
        )

    def _find_unpacked_source_root(self):
        """
        Return the directory in hypodd_src that looks like the HypoDD source.
        """
        unpack_dir = self.paths["hypodd_unpack_dir"]
        candidates = [
            os.path.join(unpack_dir, name)
            for name in os.listdir(unpack_dir)
        ]
        for candidate in candidates:
            if not os.path.isdir(candidate):
                continue
            if (
                os.path.isdir(os.path.join(candidate, "src"))
                and os.path.exists(
                    os.path.join(candidate, "include", "hypoDD.inc")
                )
            ):
                return candidate
        msg = "Could not find unpacked HypoDD source tree in %s" % unpack_dir
        raise HypoDDCompilationError(msg)

    def configure(
        self,
        MAXEVE=3000,
        MAXDATA=2800000,
        MAXEVE0=50,
        MAXDATA0=60000,
        MAXLAY=30,
        MAXSTA=2000,
        MAXCL=200,
    ):
        """
        Configure the compilation.

        **hypoDD.inc configuration**

        The following parameters are used to configure the hypoDD.inc file. The
        default values are suitable for a medium sized problem. Adjust them if
        necessary.

        :param MAXEVE: Max number of events (must be at least the size of the
            number of events listed in the event file)
            Defaults to 3000.
        :param MAXDATA: Max number of observations (must be at least the size
            of the number of observations).
            Defaults to 2800000.
        :param MAXEVE0: Max number of events used for SVD. If only LSQR is
            used, MAXEVE0 can be set to 2 to free up memory.
            Defaults to 50.
        :param MAXDATA0: Max number of observations used for SVD. If only LSQR
            is used, MAXDATA0 can be set to 1 to free up memory.
            Defaults to 60000.
        :param MAXLAY: Max number of model layers.
            Defaults to 30.
        :param MAXSTA: Max number of stations.
            Defaults to 2000.
        :param MAXCL: Max number of clusters allowed.
            Defaults to 200.
        """
        # Set the hypodd_inc configuration.
        self.hypodd_inc_config = {
            "MAXEVE": MAXEVE,
            "MAXDATA": MAXDATA,
            "MAXEVE0": MAXEVE0,
            "MAXDATA0": MAXDATA0,
            "MAXLAY": MAXLAY,
            "MAXSTA": MAXSTA,
            "MAXCL": MAXCL,
        }

        self.is_configured = True

    def unpack_archive(self):
        """
        Unpacks the HypoDD archive to the hypodd_src subfolder in the working
        directory.
        """
        self.log("Unpacking HypoDD archive ...")

        unpack_dir = self.paths["hypodd_unpack_dir"]
        if os.path.exists(unpack_dir):
            shutil.rmtree(unpack_dir)
        os.makedirs(unpack_dir)

        tar = tarfile.open(HYPODD_ARCHIVE, "r:gz")
        tar.extractall(unpack_dir)
        self._set_source_paths(self._find_unpacked_source_root())
        self._patch_unpacked_hypodd_sources()
        self.log("Unpacking HypoDD archive done.")

    def _patch_unpacked_hypodd_sources(self):
        """
        Add diagnostic double-difference residual output to HypoDD.
        """
        source_dir = os.path.join(self.paths["make_directory"], "hypoDD")
        dtres_path = os.path.join(source_dir, "dtres.f")
        hypodd_path = os.path.join(source_dir, "hypoDD.f")

        with open(dtres_path, "r") as open_file:
            dtres_source = open_file.read()
        if "subroutine write_ddres" not in dtres_source:
            dtres_source += """

      subroutine write_ddres(fn, ndt, dt_sta, dt_dt, dt_c1, dt_c2,
     & dt_idx, dt_qual, dt_cal, dt_res, dt_wt, dt_offs)

      implicit none

      include'hypoDD.inc'

      character*(*) fn
      integer ndt
      character dt_sta(MAXDATA)*7
      real dt_dt(MAXDATA)
      integer dt_c1(MAXDATA)
      integer dt_c2(MAXDATA)
      integer dt_idx(MAXDATA)
      real dt_qual(MAXDATA)
      real dt_cal(MAXDATA)
      real dt_res(MAXDATA)
      real dt_wt(MAXDATA)
      real dt_offs(MAXDATA)

      integer i
      integer iunit
      logical lexist

      call freeunit(iunit)
      inquire(file=fn,exist=lexist)
      open(iunit,file=fn,status='unknown',position='append')
      if(.not.lexist) write(iunit,'(a)')
     &'# STA OBS_S CALC_S RES_S C1 C2 IDX QUAL WT OFFS'
      write(iunit,'(a7,1x,f12.7,1x,f12.7,1x,f12.7,1x,
     & i9,1x,i9,1x,i1,1x,f9.4,1x,f11.6,1x,f8.1)')
     & (dt_sta(i),dt_dt(i),dt_cal(i),dt_res(i),dt_c1(i),dt_c2(i),
     & dt_idx(i),dt_qual(i),dt_wt(i),dt_offs(i),i=1,ndt)
      close(iunit)

      end

      subroutine write_ttimes(fn, nsrc, src_cusp, nsta, sta_lab,
     & tmp_ttp, tmp_tts)

      implicit none

      include'hypoDD.inc'

      character*(*) fn
      integer nsrc
      integer src_cusp(MAXEVE)
      integer nsta
      character sta_lab(MAXSTA)*7
      real tmp_ttp(MAXSTA,MAXEVE)
      real tmp_tts(MAXSTA,MAXEVE)

      integer i
      integer j
      integer iunit
      logical lexist

      call freeunit(iunit)
      inquire(file=fn,exist=lexist)
      open(iunit,file=fn,status='unknown',position='append')
      if(.not.lexist) write(iunit,'(a)')'# CUSP STA TTP_S TTS_S'
      do j=1,nsrc
         write(iunit,'(i9,1x,a7,1x,f12.7,1x,f12.7)')
     &   (src_cusp(j),sta_lab(i),tmp_ttp(i,j),tmp_tts(i,j),i=1,nsta)
      enddo
      close(iunit)

      end
"""
            with open(dtres_path, "w") as open_file:
                open_file.write(dtres_source)

        with open(hypodd_path, "r") as open_file:
            hypodd_source = open_file.read()
        if "hypoDD.initial.res" not in hypodd_source:
            initial_marker = (
                "       call resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,\n"
                "     & rms_cc,rms_ct,rms_cc0,rms_ct0,\n"
                "     & rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,\n"
                "     &              resvar1)\n"
                "      endif"
            )
            initial_patch = (
                "       call resstat(log,idata,ndt,nev,dt_res,dt_wt,dt_idx,\n"
                "     & rms_cc,rms_ct,rms_cc0,rms_ct0,\n"
                "     & rms_ccold,rms_ctold,rms_cc0old,rms_ct0old,\n"
                "     &              resvar1)\n"
                "       call write_ddres('hypoDD.initial.res',ndt,dt_sta,dt_dt,\n"
                "     & dt_c1,dt_c2,dt_idx,dt_qual,dt_cal,dt_res,dt_wt,dt_offs)\n"
                "       call write_ttimes('hypoDD.initial.tt',nsrc,src_cusp,nsta,\n"
                "     & sta_lab,tmp_ttp,tmp_tts)\n"
                "      endif"
            )
            if initial_marker not in hypodd_source:
                msg = "Could not patch initial HypoDD residual diagnostics."
                raise HypoDDCompilationError(msg)
            hypodd_source = hypodd_source.replace(
                initial_marker, initial_patch, 1
            )

        if "hypoDD.final.res" not in hypodd_source:
            final_marker = (
                "c--- update origin time (this is only done for final output!!)\n"
                "600   continue\n"
                "      write(*,'(/,\"writing out results ...\")')"
            )
            final_patch = (
                "c--- update origin time (this is only done for final output!!)\n"
                "600   continue\n"
                "      write(*,'(/,\"writing out results ...\")')\n"
                "\n"
                "c--- recompute final full-ray residuals for diagnostic output:\n"
                "      if(imod.eq.0.or.imod.eq.1) then\n"
                "          call partials(fn_srcpar,\n"
                "     &     nsrc,src_cusp,src_lat,src_lon,src_dep,\n"
                "     &     nsta,sta_lab,sta_lat,sta_lon,sta_elv,\n"
                "     &     mod_nl,mod_ratio,mod_v,mod_top,\n"
                "     &     tmp_ttp,tmp_tts,\n"
                "     &     tmp_xp,tmp_yp,tmp_zp,tmp_xs,tmp_ys,tmp_zs)\n"
                "      elseif(imod.eq.5) then\n"
                "          call partials_1dsr(fn_srcpar,\n"
                "     &     nsrc,src_cusp,src_lat,src_lon,src_dep,\n"
                "     &     nsta,sta_lab,sta_lat,sta_lon,sta_elv,\n"
                "     &     mod_nl,mod_ratio,mod_v,mod_top,\n"
                "     &     tmp_ttp,tmp_tts,\n"
                "     &     tmp_xp,tmp_yp,tmp_zp,tmp_xs,tmp_ys,tmp_zs)\n"
                "      elseif(imod.eq.4) then\n"
                "          call partials_1dmm(log,fn_srcpar,fn_mod1d,\n"
                "     &     nsrc,src_cusp,src_lat,src_lon,src_dep,\n"
                "     &     nsta,sta_lab,sta_lat,sta_lon,sta_elv,sta_mod,\n"
                "     &     mod_nl,mod_ratio,mod_v,mod_top,iter,\n"
                "     &     tmp_ttp,tmp_tts,\n"
                "     &     tmp_xp,tmp_yp,tmp_zp,tmp_xs,tmp_ys,tmp_zs)\n"
                "      elseif(imod.eq.9) then\n"
                "          do i=1,nsta\n"
                "             do j=1,nsrc\n"
                "                tmp_ttp(i,j)= -999\n"
                "             enddo\n"
                "          enddo\n"
                "          do k=1,ndt\n"
                "             do j=1,nsrc\n"
                "                if(dt_c1(k).eq.src_cusp(j)) tmp_ttp(dt_ista(k),j)= 0\n"
                "                if(dt_c2(k).eq.src_cusp(j)) tmp_ttp(dt_ista(k),j)= 0\n"
                "             enddo\n"
                "          enddo\n"
                "          call partials_3d(fn_srcpar,\n"
                "     &     nsrc,src_cusp,src_lat,src_lon,src_dep,\n"
                "     &     nsta,sta_lab,sta_lat,sta_lon,sta_elv,\n"
                "     &     mod_nl,mod_ratio,mod_v,mod_top,rot_3d,ipha3d,\n"
                "     &     tmp_ttp,tmp_tts,\n"
                "     &     tmp_xp,tmp_yp,tmp_zp,\n"
                "     &     tmp_xs,tmp_ys,tmp_zs)\n"
                "      endif\n"
                "      call dtres(log,ndt,MAXSTA,nsrc,\n"
                "     & dt_dt,dt_idx,\n"
                "     & dt_ista,dt_ic1,dt_ic2,\n"
                "     & src_cusp,src_t,tmp_ttp,tmp_tts,\n"
                "     & dt_cal,dt_res)\n"
                "      call write_ddres('hypoDD.final.res',ndt,dt_sta,dt_dt,\n"
                "     & dt_c1,dt_c2,dt_idx,dt_qual,dt_cal,dt_res,dt_wt,dt_offs)\n"
                "      call write_ttimes('hypoDD.final.tt',nsrc,src_cusp,nsta,\n"
                "     & sta_lab,tmp_ttp,tmp_tts)"
            )
            if final_marker not in hypodd_source:
                msg = "Could not patch final HypoDD residual diagnostics."
                raise HypoDDCompilationError(msg)
            hypodd_source = hypodd_source.replace(final_marker, final_patch, 1)

        with open(hypodd_path, "w") as open_file:
            open_file.write(hypodd_source)

    def make(self):
        if self.is_configured is not True:
            msg = "Compiler object need to be configured first."
            raise HypoDDCompilationError(msg)
        # Unpack the archive.
        self.unpack_archive()
        # Create the hypoDD_inc file.
        self.hypodd_inc_file = self.create_hypoDD_inc_file()
        # Check the current HypoDD compilation (if any).
        if self.is_current_hypodd_compilation_valid() is True:
            shutil.rmtree(self.paths["hypodd_unpack_dir"])
            self.log("Current compilation is up to date.")
            return
        # Finally compile it.
        self.compile_hypodd()
        # Cleanup.
        shutil.rmtree(self.paths["hypodd_unpack_dir"])

    def create_hypoDD_inc_file(self):
        """
        HypoDD uses static allocation and thus oftentimes has to be recompiled
        to suit a new problem size. The hypoDD.inc file is usually the only
        file that has to be changed. This is handled in this method.

        The default parameters are suitable for a medium sized problem.

        :return: A string containing the whole file.

        Original documentation:
        hypoDD.inc: Stores parameters that define array dimensions in hypoDD.
            Modify to fit size of problem and available computer memory.  If 3D
            raytracing is used, also set model parameters in vel3d.inc.

        Parameter Description:
        MAXEVE:   Max number of events (must be at least the size of the number
                  of events listed in the event file)
        MAXDATA:  Max number of observations (must be at least the size of the
                  number of observations).
        MAXEVE0:  Max number of events used for SVD. If only LSQR is used,
                  MAXEVE0 can be set to 2 to free up memory.
        MAXDATA0: Max number of observations used for SVD. If only LSQR is
            used, MAXDATA0 can be set to 1 to free up memory.
        MAXLAY:   Max number of model layers.
        MAXSTA:   Max number of stations.
        MAXCL:    Max number of clusters allowed.
        """
        # Do not mess with the indentation as it is important for Fortran77.
        hypoDD_inc = """
      integer*4 MAXEVE, MAXLAY, MAXDATA, MAXSTA, MAXEVE0, MAXDATA0
      integer*4 MAXCL
      parameter(MAXEVE   = {MAXEVE},
     &          MAXDATA  = {MAXDATA},
     &          MAXEVE0  = {MAXEVE0},
     &          MAXDATA0 = {MAXDATA0},
     &          MAXLAY   = {MAXLAY},
     &          MAXSTA   = {MAXSTA},
     &          MAXCL    = {MAXCL})""".format(
            MAXEVE=self.hypodd_inc_config["MAXEVE"],
            MAXDATA=self.hypodd_inc_config["MAXDATA"],
            MAXEVE0=self.hypodd_inc_config["MAXEVE0"],
            MAXDATA0=self.hypodd_inc_config["MAXDATA0"],
            MAXLAY=self.hypodd_inc_config["MAXLAY"],
            MAXSTA=self.hypodd_inc_config["MAXSTA"],
            MAXCL=self.hypodd_inc_config["MAXCL"],
        )
        # Remove the leading empty line.
        hypoDD_inc = hypoDD_inc[1:]
        return hypoDD_inc

    def is_current_hypodd_compilation_valid(self):
        """
        Returns True if the current compilation is ok, False otherwise. False
        should always trigger a new compilation.
        """
        # If the binary dir does not exist return False.
        if not os.path.exists(self.paths["binary_dir"]):
            return False
        # If the three file do not exist return False.
        if (
            not os.path.exists(self.paths["hypoDD_binary"])
            or not os.path.exists(self.paths["ph2dt_binary"])
            or not os.path.exists(self.paths["old hypoDD.inc file"])
            or not os.path.exists(self.paths["diagnostic_patch_stamp"])
        ):
            return False
        with open(self.paths["diagnostic_patch_stamp"], "r") as open_file:
            patch_version = open_file.read().strip()
        if patch_version != HYPODD_DIAGNOSTIC_PATCH_VERSION:
            return False
        # Check if the newly created hypoDD.inc file is identical to the old
        # one.
        with open(self.paths["old hypoDD.inc file"], "r") as open_file:
            old_hypodd_file = open_file.read()
        if old_hypodd_file != self.hypodd_inc_file:
            return False
        return True

    def compile_hypodd(self):
        """
        Actually compiles HypoDD.
        """
        # Replace hypoDD.inc file with the custom one.
        os.remove(self.paths["hypoDD.inc"])
        with open(self.paths["hypoDD.inc"], "w") as open_file:
            open_file.write(self.hypodd_inc_file)
        # Compile it.
        self.log("Compiling HypoDD ...")
        sub = subprocess.Popen(
            "make",
            cwd=self.paths["make_directory"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.log(sub.stdout.read())
        retcode = sub.wait()
        if retcode != 0:
            msg = "Problem compiling HypoDD."
            raise HypoDDCompilationError(msg)
        # Check if the output files have been created.
        if not os.path.exists(
            self.paths["compiled_hypodd_binary"]
        ) or not os.path.exists(self.paths["compiled_ph2dt_binary"]):
            msg = "The binary output files could not be found."
            raise HypoDDCompilationError(msg)
        # Move the binary files and the hypoDD.inc file.
        shutil.move(
            self.paths["compiled_hypodd_binary"], self.paths["hypoDD_binary"]
        )
        shutil.move(
            self.paths["compiled_ph2dt_binary"], self.paths["ph2dt_binary"]
        )
        shutil.move(
            self.paths["hypoDD.inc"], self.paths["old hypoDD.inc file"]
        )
        with open(self.paths["diagnostic_patch_stamp"], "w") as open_file:
            open_file.write(HYPODD_DIAGNOSTIC_PATCH_VERSION)
        self.log("Compiling HypoDD done.")
