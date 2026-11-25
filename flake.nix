{
  description = "SpikeInterface and Lupin for Linux x86-64 (CPU, AMD ROCm, NVIDIA CUDA)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    harbor-py = {
      url = "github:caniko/harbor-py";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    harbor-py,
    ...
  }: let
    system = "x86_64-linux";
    inherit (nixpkgs) lib;
    py = harbor-py.lib;
    defaultPkgs = nixpkgs.legacyPackages.${system};
    variants = ["cpu" "amd" "nvidia"];
    # Optional NVSHMEM transports; Lupin does not use distributed CUDA.
    nvshmemOptionalLibs = ["liboshmem.so.40" "libmpi.so.40" "libfabric.so.1" "libpmix.so.2" "libucs.so.0" "libucp.so.0"];

    mkEnv = {
      pkgs ? defaultPkgs,
      name,
      extras ? [],
    }: let
      python = pkgs.python313;
      site = python.sitePackages;
      baseLibs = [pkgs.stdenv.cc.cc.lib pkgs.zlib];
      rocmLibs = baseLibs ++ [pkgs.zstd pkgs.xz pkgs.bzip2 pkgs.numactl];
      selected = lib.filter (variant: builtins.elem variant extras) variants;
    in
      assert lib.assertMsg (pkgs.stdenv.hostPlatform.system == system) "SpikeInterface's uv lock requires ${system}";
      assert lib.assertMsg (builtins.length selected <= 1) "The cpu, amd, and nvidia extras are mutually exclusive";
      assert lib.assertMsg (
        !(builtins.elem "lupin" extras) || builtins.length selected == 1
      ) "Select exactly one of cpu, amd, or nvidia with the lupin extra";
        py.mkUvVirtualEnv {
          inherit pkgs python name;
          workspaceRoot = ./nix;
          dependencies.spikeinterface-nix = extras ++ lib.optional (selected == []) "base";
          pyprojectOverrides = final: prev: let
            buildWith = package: backend:
              prev.${package}.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {${backend} = [];};
              });
            addLibs = package: libs:
              prev.${package}.overrideAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ libs;
              });
            # The CUDA wheels share the nvidia namespace, but each lives in
            # its own store path. Resolve DT_NEEDED against patched siblings.
            cudaWheels = {
              cublas = [];
              cuda-cupti = [];
              cuda-nvrtc = [];
              cuda-runtime = [];
              cudnn = ["cublas"];
              cufft = ["nvjitlink"];
              cufile = [];
              curand = [];
              cusolver = ["cublas" "cusparse" "nvjitlink"];
              cusparse = ["nvjitlink"];
              cusparselt = [];
              nccl = [];
              nvjitlink = [];
              nvshmem = ["cuda-runtime"];
              nvtx = [];
            };
            cudaPackage = wheel: "nvidia-${wheel}-cu12";
            patchCuda = package: localLibDir: siblings:
              prev.${package}.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.autoAddDriverRunpath];
                buildInputs =
                  (old.buildInputs or [])
                  ++ baseLibs
                  ++ lib.optional (builtins.elem package ["nvidia-cufile-cu12" "nvidia-nvshmem-cu12"]) pkgs.rdma-core;
                # autoPatchelf resolves DT_NEEDED, not libraries opened via dlopen.
                appendRunpaths = (old.appendRunpaths or []) ++ ["${placeholder "out"}/${site}/${localLibDir}"];
                autoPatchelfIgnoreMissingDeps =
                  (old.autoPatchelfIgnoreMissingDeps or [])
                  ++ lib.optional (package == "torch") "libcuda.so.1"
                  ++ lib.optionals (package == "nvidia-nvshmem-cu12") nvshmemOptionalLibs;
                preFixup =
                  (old.preFixup or "")
                  + lib.concatMapStrings (wheel: ''
                    addAutoPatchelfSearchPath "${final.${cudaPackage wheel}}/${site}/nvidia"
                  '')
                  siblings;
              });
          in
            {
              spikeinterface = buildWith "spikeinterface" "setuptools";
              neo = buildWith "neo" "setuptools";
              probeinterface = buildWith "probeinterface" "hatchling";
              numba = addLibs "numba" (baseLibs ++ [pkgs.tbb]);
              llvmlite = addLibs "llvmlite" baseLibs;
            }
            // lib.optionalAttrs (builtins.elem "amd" extras) {
              torch = (addLibs "torch" rocmLibs).overrideAttrs (old: {
                # Use the wheel's ROCm libraries, not nixpkgs' ROCm runtime.
                appendRunpaths = (old.appendRunpaths or []) ++ ["${placeholder "out"}/${site}/torch/lib"];
              });
              triton-rocm = addLibs "triton-rocm" rocmLibs;
            }
            // lib.optionalAttrs (builtins.elem "nvidia" extras) (
              lib.mapAttrs' (
                wheel: siblings:
                  lib.nameValuePair (cudaPackage wheel) (
                    patchCuda (cudaPackage wheel) "nvidia/${builtins.replaceStrings ["-"] ["_"] wheel}/lib" siblings
                  )
              )
              cudaWheels
              // {
                torch = patchCuda "torch" "torch/lib" (builtins.attrNames cudaWheels);
                triton = (addLibs "triton" baseLibs).overrideAttrs (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.autoAddDriverRunpath];
                });
              }
            );
        };

    pkgs = defaultPkgs;
    packages = let
      base = mkEnv {name = "spikeinterface";};
      lupin = lib.listToAttrs (map (variant:
        lib.nameValuePair "lupin-${variant}" (mkEnv {
          name = "lupin-${variant}";
          extras = ["lupin" variant];
        }))
      variants);
    in
      lupin
      // {
        spikeinterface = base;
        default = base;
        lupin-rocm = lupin.lupin-amd;
        lupin-cuda = lupin.lupin-nvidia;
        lupin-nvidia-check =
          pkgs.runCommand "lupin-nvidia-check" {
            nativeBuildInputs = [pkgs.binutils pkgs.glibc.bin];
          } ''
            unset LD_LIBRARY_PATH
            ${lupin.lupin-nvidia}/bin/python ${./nix/smoke.py} nvidia-offline
            site=${lupin.lupin-nvidia}/${pkgs.python313.sitePackages}
            find -L "$site"/{torch,nvidia,triton} -type f -print0 > files
             scanned=0
             linked=0
             driver=0
             while IFS= read -r -d "" elf; do
               # readelf also accepts .a archives, which have no runtime loader.
               IFS= read -r -N 4 magic < "$elf" && [[ "$magic" == $'\x7fELF' ]] || continue
               dynamic=$(readelf -d "$elf")
              scanned=$((scanned + 1))
              [[ "$dynamic" == *"Dynamic section"* ]] || continue
              # Driver sonames can appear in DT_NEEDED or as dlopen strings.
              if grep -aFq -e libcuda.so -e libnvidia-ml.so "$elf"; then
                runpath=$(awk -F '[][]' '/\(RUNPATH\)/ {print $2}' <<< "$dynamic")
                if [[ ":$runpath:" != *:/run/opengl-driver/lib:* ]]; then
                  echo "Missing NVIDIA driver RUNPATH: $elf" >&2
                  exit 1
                fi
                driver=$((driver + 1))
              fi
              [[ "$dynamic" == *"(NEEDED)"* ]] || continue
              links=$(ldd "$elf" 2>"$TMPDIR/ldd.err") || {
                cat "$TMPDIR/ldd.err" >&2
                echo "$elf: $links" >&2
                exit 1
              }
              # ldd reports transitive dependencies too; the driver is absent in the sandbox.
              allowed="libcuda.so.1"
              if [[ "$elf" == "$site"/nvidia/nvshmem/lib/nvshmem_bootstrap_* ||
                    "$elf" == "$site"/nvidia/nvshmem/lib/nvshmem_transport_* ]]; then
                allowed+=" ${lib.concatStringsSep " " nvshmemOptionalLibs}"
              fi
              unexpected=$(awk -v allowed=" $allowed " '/=> not found/ && !index(allowed, " " $1 " ")' <<< "$links")
               if [[ -n "$unexpected" ]]; then
                 echo "$elf: $unexpected" >&2
                 exit 1
               fi
              linked=$((linked + 1))
            done < files
            test "$linked" -gt 0 && test "$driver" -gt 0 || exit 1
            echo "Scanned $scanned ELF files; ldd checked $linked, NVIDIA driver RUNPATH checked $driver" | tee "$out"
          '';
      };

    shells = py.mkUvDevShells {
      inherit pkgs;
      extras = variants;
      defaultExtra = "cpu";
      mkShellArgs = variant: {
        autoSync = false;
        opencodeLsp.enable = false;
        basePackages = [pkgs.ruff];
        shellHookSuffix = ''
          export PATH="${packages."lupin-${variant}"}/bin:$PATH"
        '';
      };
    };
  in {
    lib.${system} = {inherit mkEnv;};
    packages.${system} = packages;
    devShells.${system} =
      shells
      // {
        rocm = shells.amd;
        cuda = shells.nvidia;
      };
    checks.${system}.lupin-cpu = pkgs.runCommand "lupin-cpu-smoke" {} ''
      set -o pipefail
      export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
      export NUMBA_CACHE_DIR="$TMPDIR/numba"
      ${packages.lupin-cpu}/bin/python ${./nix/smoke.py} cpu | tee "$out"
    '';
  };
}
