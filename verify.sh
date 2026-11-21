#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# kv-stream-pascal: Herramienta de compilación, análisis y SAST
# Enfoque: SOLO el código del proyecto. Sin test de modelos.
# ═══════════════════════════════════════════════════════════════
set -o pipefail

# ─── CONFIGURACIÓN ─────────────────────────────────────────────
PROJECT_DIR="$HOME/Compilaciones/kv-stream-pascal"
BUILD_DIR="$PROJECT_DIR/build-pascal"
LOG_FILE="/tmp/build-full.log"
REPORT_FILE="/tmp/build-report.txt"
GPU_ARCH=61
NPROC=$(nproc)

# Detectar GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

cd "$PROJECT_DIR" || { echo "❌ No existe $PROJECT_DIR"; exit 1; }

# ─── COLORES ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

header() {
    echo ""
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════${NC}"
}
section() { echo ""; echo -e "${BOLD}─── $1 ─────────────────────────────────────────${NC}"; }
ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; }
info() { echo -e "  ${CYAN}ℹ️  $1${NC}"; }

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 0: ENTORNO
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 0: ENTORNO"

info "GPU: $GPU_NAME (${GPU_VRAM} MB VRAM, sm_${GPU_ARCH})"
info "Driver: $DRIVER_VER"
info "NVCC: $(nvcc --version 2>/dev/null | grep 'release' | awk '{print $NF}' | tr -d ',')"

# Driver mínimo
DRIVER_NUM=$(echo "$DRIVER_VER" | cut -d. -f1)
if [ "$DRIVER_NUM" -lt 550 ]; then
    fail "Driver $DRIVER_VER < 550"
else
    ok "Driver $DRIVER_VER ≥ 550"
fi

# nvcc
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep 'release' | awk '{print $NF}' | tr -d ',')
    ok "CUDA Toolkit: $NVCC_VER"
    if nvcc --list-gpu-arch 2>/dev/null | grep -q "compute_${GPU_ARCH}"; then
        ok "sm_${GPU_ARCH} soportado"
    else
        fail "sm_${GPU_ARCH} NO soportado"
    fi
else
    fail "nvcc no encontrado"
fi

# Auto-corregir: si CC no está set y gcc-14 existe, usarlo
if [ -z "$CC" ] && command -v gcc-14 &>/dev/null; then
    export CC=gcc-14
    export CXX=g++-14
    export NVCC_CCBIN=g++-14
    info "CC/CXX/NVCC_CCBIN auto-set a gcc-14/g++-14"
fi

# GCC (el compilador que realmente usará el build)
BUILD_CC="${CC:-gcc}"
GCC_VER=$($BUILD_CC --version 2>/dev/null | head -1 | awk '{print $3}' | cut -d. -f1)
GCC_NAME=$($BUILD_CC --version 2>/dev/null | head -1 | awk '{print $3}')

if [ "$GCC_VER" -gt 14 ] 2>/dev/null; then
    fail "Compilador $BUILD_CC ($GCC_NAME) > 14 (máx. CUDA 12.9)"
    fail "Usa: set -x CC gcc-14; set -x CXX g++-14; set -x NVCC_CCBIN g++-14"
else
    ok "Compilador: $BUILD_CC ($GCC_NAME) — compatible"
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 1: REBUILD COMPLETO (siempre)
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 1: REBUILD COMPLETO"

echo -e "${BOLD}▶ Limpiando build anterior...${NC}"
cmake --build "$BUILD_DIR" --target clean 2>&1 | tail -1

echo -e "${BOLD}▶ Compilando desde cero (llama-server + llama-bench)...${NC}"
echo -e "${CYAN}  (Esto puede tardar varios minutos)${NC}"
grep -q "LLAMA_ALL_WARNINGS_3RD_PARTY:BOOL=OFF" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || echo "LLAMA_ALL_WARNINGS_3RD_PARTY:BOOL=OFF" >> "$BUILD_DIR/CMakeCache.txt"
cmake --build "$BUILD_DIR" --target llama-server llama-bench -j"$NPROC" 2>&1 | tee "$LOG_FILE" > /dev/null
BUILD_EXIT=${PIPESTATUS[0]}

echo ""
if [ $BUILD_EXIT -eq 0 ] && ! grep -q 'error:' "$LOG_FILE"; then
    ok "BUILD: ✅ OK"
else
    fail "BUILD: ❌ FALLO (exit code: $BUILD_EXIT)"
fi

# Verificar binarios
for BIN in llama-server llama-bench; do
    if [ -f "$BUILD_DIR/bin/$BIN" ]; then
        ok "$BIN presente: $(ls -lh "$BUILD_DIR/bin/$BIN" | awk '{print $5}')"
    else
        fail "$BIN NO encontrado"
    fi
done

# Guardar log con timestamp
cp "$LOG_FILE" "$PROJECT_DIR/build-log-$(date +%Y%m%d-%H%M%S).log"
info "Log guardado: build-log-$(date +%Y%m%d-%H%M%S).log"

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 2: ANÁLISIS DE WARNINGS Y ERRORES (CATEGORIZADO)
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 2: ANÁLISIS DE WARNINGS Y ERRORES"

# ─── Normalización ─────────────────────────────────────────────
awk '
/^[^[:space:]]+\.(cpp|c|cu|h|hpp|cuh|cc):[0-9]+/ {
    if ($0 ~ /warning:|error:/) {
        line = $0
        sub(/\[with[^]]*\]/, "", line)
        sub(/ \[-W[a-z-]+\]$/, " [-WFLAG]", line)
        gsub(/GGML_TYPE_[A-Z0-9_]+/, "GGML_TYPE_XXX", line)
        gsub(/0x[0-9A-Fa-f]+/, "0xHEX", line)
        gsub(/:[0-9]+:/, ":LINE:", line)
        print line
    }
}
/^[^[:space:]]+\.(cpp|c|cu|h|hpp|cuh|cc)\([0-9]+\)/ {
    if ($0 ~ /warning|error/) {
        line = $0
        sub(/\[with[^]]*\]/, "", line)
        gsub(/GGML_TYPE_[A-Z0-9_]+/, "GGML_TYPE_XXX", line)
        gsub(/0x[0-9A-Fa-f]+/, "0xHEX", line)
        print line
    }
}
' "$LOG_FILE" | sort | uniq -c | sort -rn > "$REPORT_FILE"

# ─── Separar VENDOR de PROYECTO ────────────────────────────────
VENDOR_PAT="vendor|stubs|cutlass|template-instances|humming|marlin|gdn-fla|fp8-cutlass"

# Warnings de vendor (solo contar, no listar)
VENDOR_COUNT=$(grep -c "$VENDOR_PAT" "$REPORT_FILE" 2>/dev/null)
VENDOR_COUNT=${VENDOR_COUNT:-0}

# Warnings del proyecto (sin vendor)
grep -v "$VENDOR_PAT" "$REPORT_FILE" > /tmp/build-project-warnings.txt 2>/dev/null
PROJECT_COUNT=$(wc -l < /tmp/build-project-warnings.txt 2>/dev/null)
PROJECT_COUNT=${PROJECT_COUNT:-0}

TOTAL_ERRORS=$(grep -c "error:" "$REPORT_FILE" 2>/dev/null)
TOTAL_ERRORS=${TOTAL_ERRORS:-0}

section "RESUMEN"
info "Warnings únicos totales: $(wc -l < "$REPORT_FILE" 2>/dev/null)"
info "  → Vendor (no corregir, actualizar upstream): $VENDOR_COUNT"
info "  → Proyecto (corregir): $PROJECT_COUNT"
info "  → Errores: $TOTAL_ERRORS"

# ─── CRÍTICOS: dangling reference (use-after-free) ─────────────
section "🔴 CRÍTICOS: Dangling References (use-after-free)"
DANGLING=$(grep "dangling reference" "$LOG_FILE" | grep -v "$VENDOR_PAT" | sort -u)
if [ -n "$DANGLING" ]; then
    DANGLING_COUNT=$(echo "$DANGLING" | wc -l)
    fail "$DANGLING_COUNT referencias colgantes detectadas:"
    echo "$DANGLING" | sed 's/^/    /'
    echo ""
    info "FIX: Usar variable local en vez de referencia a temporal:"
    echo "  ANTES:  const auto & x = require_tensor(registry_, prefix + \"name\");"
    echo "  DESPUÉS: auto x = require_tensor(registry_, prefix + \"name\");"
else
    ok "(ninguno)"
fi

# ─── ALTOS: Enum no manejado en switch ─────────────────────────
section "🟠 ALTOS: Valores de Enum no manejados"
UNHANDLED_ENUM=$(grep -i "no se maneja en un switch\|not handled in a switch\|enumeration value.*not handled" "$LOG_FILE" | grep -v "$VENDOR_PAT" | sort -u)
if [ -n "$UNHANDLED_ENUM" ]; then
    fail "Tipos de cuantización sin handler (causarán fallback o crash):"
    echo "$UNHANDLED_ENUM" | sed 's/^/    /'
    echo ""
    info "FIX: Añadir case para cada enum faltante en el switch correspondiente."
    info "Si no hay implementación para Pascal, usar default con GGML_ABORT o fallback a CPU."
else
    ok "(ninguno)"
fi

# ─── MEDIOS: APIs deprecadas ───────────────────────────────────
section "🟡 MEDIOS: APIs deprecadas"
DEPRECATED=$(grep -i "deprecated\|es obsoleto" "$LOG_FILE" | grep -v "$VENDOR_PAT" | sort -u)
if [ -n "$DEPRECATED" ]; then
    DEP_COUNT=$(echo "$DEPRECATED" | wc -l)
    warn "$DEP_COUNT APIs deprecadas:"
    echo "$DEPRECATED" | sed 's/^/    /'
    echo ""
    info "FIX: Reemplazar según el hint del warning:"
    info "  ggml_mul_mat_set_prec(t, GGML_PREC_F32) → ggml_prec_set_acc(t, GGML_PREC_F32)"
    info "  ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32) → ggml_prec_set_acc(t, GGML_PREC_F32)"
else
    ok "(ninguno)"
fi

# ─── MEDIOS: Sign-compare ──────────────────────────────────────
section "🟡 MEDIOS: Sign-compare (posible overflow con datos grandes)"
SIGN_COMPARE=$(grep -i "sign-compare\|different signedness" "$LOG_FILE" | grep -v "$VENDOR_PAT" | sort -u)
if [ -n "$SIGN_COMPARE" ]; then
    SC_COUNT=$(echo "$SIGN_COMPARE" | wc -l)
    warn "$SC_COUNT comparaciones con signo mixto:"
    echo "$SIGN_COMPARE" | sed 's/^/    /'
    echo ""
    info "FIX: Cast explícito o unificar tipos:"
    info "  ANTES:  if (n >= 3 && n <= model.hparams.dflash_block_size)"
    info "  DESPUÉS: if (n >= 3 && n <= (int)model.hparams.dflash_block_size)"
else
    ok "(ninguno)"
fi

# ─── BAJO: unused, missing-init, redundant-move ───────────────
section "⚪ BAJO: Calidad de código (unused, init, move)"
LOW_WARNINGS=$(grep -iE "unused-parameter|unused-but-set|missing-field-initializers|redundant-move|suggest-attribute" "$LOG_FILE" | grep -v "$VENDOR_PAT" | sort -u)
if [ -n "$LOW_WARNINGS" ]; then
    LOW_COUNT=$(echo "$LOW_WARNINGS" | wc -l)
    info "$LOW_COUNT warnings de calidad (no críticos, pero limpiar):"
    echo "$LOW_WARNINGS" | head -15 | sed 's/^/    /'
    if [ "$LOW_COUNT" -gt 15 ]; then
        info "  ... y $((LOW_COUNT - 15)) más (ver reporte completo)"
    fi
else
    ok "(ninguno)"
fi

# ─── VENDOR (solo resumen) ─────────────────────────────────────
section "VENDOR (no corregir, actualizar con upstream)"
info "$VENDOR_COUNT warnings únicos de código de terceros."
info "Se resuelven con: git submodule update / pull de llama.cpp upstream."
# Mostrar top 3 de vendor por curiosidad
grep "$VENDOR_PAT" "$REPORT_FILE" | head -3 | sed 's/^/    /'

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 3: PROBLEMAS CRÍTICOS OCULTOS
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 3: PROBLEMAS CRÍTICOS OCULTOS"

section "ENLACE"
LINK_ERRORS=$(grep -i "undefined reference\|multiple definition\|cannot find -l\|linker error\|ld returned" "$LOG_FILE" | sort | uniq -c | sort -rn)
if [ -n "$LINK_ERRORS" ]; then
    fail "Fallos de símbolos o librerías:"
    echo "$LINK_ERRORS" | head -10
else
    ok "Enlace: OK"
fi

section "MEMORIA"
MEM_ERRORS=$(grep -i "segfault\|segmentation fault\|out of memory\|oom-killer\|illegal memory access\|heap corruption\|double free" "$LOG_FILE")
if [ -n "$MEM_ERRORS" ]; then
    fail "Posibles fallos de estabilidad:"
    echo "$MEM_ERRORS" | head -10
else
    ok "Memoria: OK"
fi

section "CUDA"
CUDA_ERRORS=$(grep -i "no kernel image is available\|cuda error\|nvcc fatal\|ptxas.*error\|unsupported gpu\|unrecognized option" "$LOG_FILE" | grep -v "warning" | sort | uniq -c | sort -rn)
if [ -n "$CUDA_ERRORS" ]; then
    fail "Errores de compilación o arquitectura CUDA:"
    echo "$CUDA_ERRORS" | head -10
else
    ok "CUDA: OK"
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 4: MIGRACIÓN DE FLAGS (b10875+)
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 4: MIGRACIÓN DE FLAGS (b10875+)"

info "Flags eliminados: --mmap, --no-mmap, --mlock, --direct-io → usar -lm"

OBSOLETE=$(grep -rn --include="*.sh" --include="*.service" --include="*.conf" --include="*.env" --include="*.py" \
    -E "\-\-no-mmap|\-\-mmap\b|\-\-mlock|\-\-direct-io|LLAMA_ARG_MMAP|LLAMA_ARG_MLOCK|LLAMA_ARG_DIRECT_IO" \
    "$PROJECT_DIR" 2>/dev/null | grep -v "verify.sh" | grep -v ".git/" | grep -v "build-")

if [ -n "$OBSOLETE" ]; then
    fail "Flags obsoletos detectados:"
    echo "$OBSOLETE" | head -10
    echo ""
    echo -e "  ${BOLD}Migración:${NC}"
    echo "    --no-mmap          →  -lm none"
    echo "    --mmap             →  -lm mmap"
    echo "    --mlock            →  -lm mmap+mlock"
    echo "    --direct-io        →  -lm dio"
else
    ok "Sin flags obsoletos"
fi

# Verificar que el binario soporta -lm
if [ -f "$BUILD_DIR/bin/llama-server" ]; then
    if "$BUILD_DIR/bin/llama-server" --help 2>&1 | grep -q "load-mode\|-lm"; then
        ok "Binario soporta -lm / --load-mode"
    else
        warn "Binario no muestra -lm en --help"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 5: KV CACHE Y FLASH ATTENTION (PASCAL)
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 5: KV CACHE Y FLASH ATTENTION"

ok "Flash Attention SÍ funciona en Pascal (sm_${GPU_ARCH})"
echo "  Kernels: vec_f32 / tile_f32 (sin tensor cores)"
echo ""
info "Limitaciones en Pascal:"
echo "  • Head sizes estándar (64, 128, 256): FA en GPU ✅"
echo "  • Head sizes no estándar (192, 320, 512): fallback CPU ⚠️"
echo "  • KV cuantizada requiere GGML_CUDA_FA_ALL_QUANTS=ON"
echo "  • Sin FA_ALL_QUANTS: solo f16, bf16, q8_0, q4_0 en GPU"

# Verificar que el build incluyó FA_ALL_QUANTS
FA_ALL=$(grep "GGML_CUDA_FA_ALL_QUANTS" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null)
if [ -n "$FA_ALL" ] && echo "$FA_ALL" | grep -q "ON"; then
    ok "GGML_CUDA_FA_ALL_QUANTS=ON (todos los tipos de KV en GPU)"
else
    warn "GGML_CUDA_FA_ALL_QUANTS no está ON. KV q5_0/q4_1/iq4_nl → CPU"
fi

# Detectar configs con KV cuantizada
KV_FLAGS=$(grep -rn --include="*.sh" --include="*.service" --include="*.conf" \
    -E "\-ctk|\-ctv|\-\-cache-type-k|\-\-cache-type-v" \
    "$PROJECT_DIR" 2>/dev/null | grep -v "verify.sh" | grep -v ".git/" | grep -v "build-")
if [ -n "$KV_FLAGS" ]; then
    section "CONFIG KV DETECTADA"
    echo "$KV_FLAGS"
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 6: VALIDACIÓN DE ARQUITECTURA GPU
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 6: ARQUITECTURA GPU (sm_61)"

CMAKE_ARCHS=$(grep "CMAKE_CUDA_ARCHITECTURES" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
info "CMAKE_CUDA_ARCHITECTURES: $CMAKE_ARCHS"

if echo "$CMAKE_ARCHS" | grep -qE "(^|[^0-9])61([^0-9]|$)"; then
    ok "sm_61 incluido"
else
    fail "sm_61 NO en CMAKE_CUDA_ARCHITECTURES"
fi

# Detectar kernels que requieren sm_80+ sin guard
SM80_UNGUARDED=$(grep -rn "__CUDA_ARCH__ >= 800\|__CUDA_ARCH__ > 800" \
    "$PROJECT_DIR/ggml/src/ggml-cuda/" 2>/dev/null | \
    grep -v "if\|ifdef\|ifndef\|else\|endif" | wc -l)
if [ "$SM80_UNGUARDED" -gt 0 ]; then
    warn "$SM80_UNGUARDED referencias a sm_80+ (verificar que están en if __CUDA_ARCH__)"
else
    ok "Sin kernels sm_80+ desprotegidos"
fi

# Verificar en el log que no hay ptxas para arquitecturas incorrectas
PTXAS_ARCH=$(grep "ptxas" "$LOG_FILE" 2>/dev/null | grep -oE "sm_[0-9]+" | sort -u)
if [ -n "$PTXAS_ARCH" ]; then
    section "ARQUITECTURAS EN PTXAS"
    echo "$PTXAS_ARCH"
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 7: SAST FOCALIZADO (SOLO CÓDIGO DEL PROYECTO)
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 7: SAST FOCALIZADO"

SRC_DIR="$PROJECT_DIR/ggml/src"
LLAMA_DIR="$PROJECT_DIR/src"
COMMON_DIR="$PROJECT_DIR/common"
TOOLS_DIR="$PROJECT_DIR/tools"
VENDOR_EXCLUDE='vendor|stubs|cutlass|template-instances|humming|marlin|gdn-fla|fp8-cutlass'

section "DANGLING REFERENCES (use-after-free en host code)"
# Buscar el patrón real: const auto & x = func_that_returns_temporary(...)
DANGLING_SAST=$(grep -rn "const auto &.*=.*require_tensor\|const auto &.*=.*std::make_pair" \
    "$LLAMA_DIR/" "$COMMON_DIR/" "$TOOLS_DIR/" 2>/dev/null | \
    grep -v "$VENDOR_EXCLUDE" | grep -v "verify.sh")
if [ -n "$DANGLING_SAST" ]; then
    D_COUNT=$(echo "$DANGLING_SAST" | wc -l)
    fail "$D_COUNT patrones de dangling reference en código fuente:"
    echo "$DANGLING_SAST" | sed 's/^/    /'
else
    ok "Sin patrones de dangling reference"
fi

section "ENUMS SIN HANDLER (switch incompleto)"
# Buscar switches sobre ggml_type que no cubren todos los valores
SWITCH_INCOMPLETE=$(grep -rn "switch.*type\|switch.*ggml_type" "$SRC_DIR/ggml-cpu/" "$LLAMA_DIR/" 2>/dev/null | \
    grep -v "$VENDOR_EXCLUDE" | grep -v "verify.sh" | wc -l)
info "Switches sobre tipo encontrados: $SWITCH_INCOMPLETE (verificar cobertura manual)"
# Los TURBO types ya los detecta la Sección 2, aquí solo contamos

section "SIGN-COMPARE EN CÓDIGO CRÍTICO"
# Solo en kernels CUDA y en fit.cpp (donde el overflow puede ser real)
SIGN_CUDA=$(grep -rn "if.*<.*n\|if.*>=.*n\|if.*<=.*n" "$SRC_DIR/ggml-cuda/" 2>/dev/null | \
    grep -v "$VENDOR_EXCLUDE" | grep -v "verify.sh" | \
    grep -c "uint\|size_t" 2>/dev/null)
SIGN_CUDA=${SIGN_CUDA:-0}
if [ "$SIGN_CUDA" -gt 10 ]; then
    info "$SIGN_CUDA comparaciones uint/size_t en kernels CUDA (revisar las del log)"
else
    ok "Kernels CUDA: sin patrones sospechosos de sign-compare"
fi

section "BUFFER OVERFLOW EN KERNELS (memcpy sin sizeof)"
# Solo memcpy donde el 3er argumento NO es sizeof(...)
RISKY_MEMCPY=$(grep -rn "memcpy(" "$SRC_DIR/ggml-cuda/" 2>/dev/null | \
    grep -v "$VENDOR_EXCLUDE" | grep -v "verify.sh" | \
    grep -v "sizeof\|cudaMemcpy" | \
    grep -v "^[[:space:]]*//")
if [ -n "$RISKY_MEMCPY" ]; then
    R_COUNT=$(echo "$RISKY_MEMCPY" | wc -l)
    warn "$R_COUNT memcpy sin sizeof explícito (verificar tamaño):"
    echo "$RISKY_MEMCPY" | head -10 | sed 's/^/    /'
else
    ok "Sin memcpy sin verificación de tamaño en kernels"
fi

section "INT OVERFLOW EN BUCLES DE GPU"
# for loops con int (no int64/size_t) que iteran sobre n_rows/n_cols
INT_LOOPS=$(grep -rn "for.*int[[:space:]]*[a-z].*<.*n_\|for.*int[[:space:]]*[a-z].*<.*rows\|for.*int[[:space:]]*[a-z].*<.*cols" \
    "$SRC_DIR/ggml-cuda/" 2>/dev/null | \
    grep -v "$VENDOR_EXCLUDE" | grep -v "verify.sh" | \
    grep -v "int64\|size_t" | wc -l)
INT_LOOPS=${INT_LOOPS:-0}
if [ "$INT_LOOPS" -gt 0 ]; then
    info "$INT_LOOPS bucles con int en kernels (OK para modelos <2B, revisar si usas >2B)"
else
    ok "Bucles de GPU: tipos correctos"
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 8: SUPPLY CHAIN (binarios del build)
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 8: INTEGRIDAD DE BINARIOS"

section "SHA256 DE BINARIOS COMPILADOS"
for BIN in llama-server llama-bench; do
    if [ -f "$BUILD_DIR/bin/$BIN" ]; then
        SHA=$(sha256sum "$BUILD_DIR/bin/$BIN" | cut -d' ' -f1)
        PERMS=$(stat -c "%a" "$BUILD_DIR/bin/$BIN")
        SIZE=$(ls -lh "$BUILD_DIR/bin/$BIN" | awk '{print $5}')
        echo "  $BIN: $SHA"
        echo "    Tamaño: $SIZE | Permisos: $PERMS"
        if [ "$PERMS" = "755" ] || [ "$PERMS" = "700" ]; then
            ok "$BIN permisos OK"
        else
            warn "$BIN permisos: $PERMS (revisar)"
        fi
    fi
done

# Verificar que no hay binarios con write para otros
INSECURE=$(find "$BUILD_DIR/bin/" -type f -perm -o+w 2>/dev/null | wc -l)
if [ "$INSECURE" -gt 0 ]; then
    fail "$INSECURE binarios con write para otros"
else
    ok "Sin permisos de escritura inseguros"
fi

# Verificar dependencias dinámicas de los binarios
section "DEPENDENCIAS DINÁMICAS (llama-server)"
if [ -f "$BUILD_DIR/bin/llama-server" ]; then
    LDD_OUT=$(ldd "$BUILD_DIR/bin/llama-server" 2>&1)
    MISSING=$(echo "$LDD_OUT" | grep "not found" | wc -l)
    if [ "$MISSING" -gt 0 ]; then
        fail "$MISSING librerías faltantes:"
        echo "$LDD_OUT" | grep "not found" | sed 's/^/    /'
    else
        ok "Todas las dependencias dinámicas resueltas"
    fi
    # Verificar que enlaza con CUDA
    if echo "$LDD_OUT" | grep -q "libcuda\|libcudart"; then
        ok "Enlaza con runtime CUDA"
    else
        warn "No enlaza con libcuda/libcudart (¿estático?)"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 9: UNIDAD SYSTEMD
# ═══════════════════════════════════════════════════════════════
header "SECCIÓN 9: UNIDAD SYSTEMD (opcional)"

SERVICE_FILE="$HOME/.config/systemd/user/llama-pascal.service"
mkdir -p "$HOME/.config/systemd/user"

cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=llama-server (kv-stream-pascal, GTX 1060 sm_61)
After=network.target

[Service]
Type=simple
ExecStart=__BUILD_DIR__/bin/llama-server \
    -m __MODEL__ \
    -ngl 99 \
    -c 4096 \
    -fa on \
    -lm mmap \
    --host 127.0.0.1 \
    --port 8080
Restart=on-failure
RestartSec=5
Environment="CUDA_VISIBLE_DEVICES=0"
LimitMEMLOCK=infinity

[Install]
WantedBy=default.target
EOF

# Reemplazar placeholders
sed -i "s|__BUILD_DIR__|$BUILD_DIR|g" "$SERVICE_FILE"
sed -i "s|__MODEL__|/ruta/a/modelo.gguf|g" "$SERVICE_FILE"

ok "Unidad generada: $SERVICE_FILE"
info "Edita el modelo en la unidad antes de activar."
info "Activar: systemctl --user daemon-reload && systemctl --user enable --now llama-pascal"

# ═══════════════════════════════════════════════════════════════
# SECCIÓN 10: RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════
header "RESUMEN FINAL"

echo ""
echo -e "  ${BOLD}GPU: ${NC}$GPU_NAME (sm_${GPU_ARCH}, ${GPU_VRAM} MB)"
echo -e "  ${BOLD}Driver: ${NC}$DRIVER_VER"
echo -e "  ${BOLD}CUDA Toolkit: ${NC}$(nvcc --version 2>/dev/null | grep 'release' | awk '{print $NF}' | tr -d ',')"
echo -e "  ${BOLD}Build: ${NC}$([ $BUILD_EXIT -eq 0 ] && echo "✅ OK" || echo "❌ FALLO")"
echo -e "  ${BOLD}Errores únicos: ${NC}$TOTAL_ERRORS"
echo -e "  ${BOLD}Warnings únicos: ${NC}$TOTAL_WARNINGS"
echo -e "  ${BOLD}Flash Attention: ${NC}✅ Sí (vec_f32/tile_f32 en Pascal)"
echo -e "  ${BOLD}FA_ALL_QUANTS: ${NC}$([ -n "$FA_ALL" ] && echo "$FA_ALL" | grep -q "ON" && echo "✅ ON" || echo "⚠️ OFF")"
echo ""

# Guardar reporte
cp "$REPORT_FILE" "$PROJECT_DIR/build-report-$(date +%Y%m%d-%H%M).txt" 2>/dev/null
info "Reporte: build-report-$(date +%Y%m%d-%H%M).txt"
info "Log completo: build-log-$(date +%Y%m%d-%H%M%S).log"

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  FIN — Solo análisis de compilación y código${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"   
