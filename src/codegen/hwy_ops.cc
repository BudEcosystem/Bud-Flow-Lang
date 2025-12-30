// =============================================================================
// Bud Flow Lang - Highway SIMD Operations Implementation
// =============================================================================
//
// This file provides portable SIMD operations using Google Highway's
// multi-target dispatch mechanism. The code is automatically compiled
// for all supported SIMD instruction sets and the best one is selected
// at runtime.
//
// =============================================================================

#include "bud_flow_lang/codegen/hwy_ops.h"

// Highway multi-target include mechanism
// foreach_target.h will include hwy_ops-inl.h once per SIMD target,
// EXCEPT for the static target. We include it again after foreach_target.h
// for the static target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy_ops-inl.h"
#include <hwy/aligned_allocator.h>
#include <hwy/foreach_target.h>  // IWYU pragma: keep
#include <hwy/highway.h>

#include <cstdlib>  // For std::aligned_alloc, std::free

#include "hwy_ops-inl.h"  // For the static target

// VQSort - vectorized quicksort (pre-compiled library, not HWY_EXPORT pattern)
#include <hwy/contrib/sort/vqsort.h>

// =============================================================================
// HWY_EXPORT declarations - must be in hwy namespace or global scope
// =============================================================================
namespace hwy {

// Priority 1: Arithmetic Operations
HWY_EXPORT(AddFloat32);
HWY_EXPORT(AddFloat64);
HWY_EXPORT(AddInt32);
HWY_EXPORT(AddInt64);
HWY_EXPORT(SubFloat32);
HWY_EXPORT(SubFloat64);
HWY_EXPORT(SubInt32);
HWY_EXPORT(SubInt64);
HWY_EXPORT(MulFloat32);
HWY_EXPORT(MulFloat64);
HWY_EXPORT(MulInt32);
HWY_EXPORT(MulInt64);
HWY_EXPORT(DivFloat32);
HWY_EXPORT(DivFloat64);
HWY_EXPORT(NegFloat32);
HWY_EXPORT(NegFloat64);
HWY_EXPORT(NegInt32);
HWY_EXPORT(NegInt64);
HWY_EXPORT(AbsFloat32);
HWY_EXPORT(AbsFloat64);
HWY_EXPORT(AbsInt32);
HWY_EXPORT(AbsInt64);

// Priority 1: FMA Operations
HWY_EXPORT(MulAddFloat32);
HWY_EXPORT(MulAddFloat64);
HWY_EXPORT(MulSubFloat32);
HWY_EXPORT(MulSubFloat64);
HWY_EXPORT(NegMulAddFloat32);
HWY_EXPORT(NegMulAddFloat64);

// Priority 1: MinMax Operations
HWY_EXPORT(MinFloat32);
HWY_EXPORT(MinFloat64);
HWY_EXPORT(MinInt32);
HWY_EXPORT(MinInt64);
HWY_EXPORT(MaxFloat32);
HWY_EXPORT(MaxFloat64);
HWY_EXPORT(MaxInt32);
HWY_EXPORT(MaxInt64);
HWY_EXPORT(ClampFloat32);
HWY_EXPORT(ClampFloat64);
HWY_EXPORT(ClampInt32);
HWY_EXPORT(ClampInt64);

// Priority 1: Math Operations
HWY_EXPORT(SqrtFloat32);
HWY_EXPORT(SqrtFloat64);
HWY_EXPORT(RsqrtFloat32);
HWY_EXPORT(RsqrtFloat64);
HWY_EXPORT(ExpFloat32);
HWY_EXPORT(ExpFloat64);
HWY_EXPORT(LogFloat32);
HWY_EXPORT(LogFloat64);
HWY_EXPORT(SinFloat32);
HWY_EXPORT(SinFloat64);
HWY_EXPORT(CosFloat32);
HWY_EXPORT(CosFloat64);
HWY_EXPORT(TanhFloat32);
HWY_EXPORT(TanhFloat64);

// Priority 1: Comparison Operations
HWY_EXPORT(EqFloat32);
HWY_EXPORT(EqFloat64);
HWY_EXPORT(EqInt32);
HWY_EXPORT(NeFloat32);
HWY_EXPORT(NeFloat64);
HWY_EXPORT(NeInt32);
HWY_EXPORT(LtFloat32);
HWY_EXPORT(LtFloat64);
HWY_EXPORT(LtInt32);
HWY_EXPORT(LeFloat32);
HWY_EXPORT(LeFloat64);
HWY_EXPORT(LeInt32);
HWY_EXPORT(GtFloat32);
HWY_EXPORT(GtFloat64);
HWY_EXPORT(GtInt32);
HWY_EXPORT(GeFloat32);
HWY_EXPORT(GeFloat64);
HWY_EXPORT(GeInt32);

// Priority 1: Reduction Operations
HWY_EXPORT(ReduceSumFloat32);
HWY_EXPORT(ReduceSumFloat64);
HWY_EXPORT(ReduceSumInt32);
HWY_EXPORT(ReduceSumInt64);
HWY_EXPORT(ReduceMinFloat32);
HWY_EXPORT(ReduceMinFloat64);
HWY_EXPORT(ReduceMinInt32);
HWY_EXPORT(ReduceMinInt64);
HWY_EXPORT(ReduceMaxFloat32);
HWY_EXPORT(ReduceMaxFloat64);
HWY_EXPORT(ReduceMaxInt32);
HWY_EXPORT(ReduceMaxInt64);
HWY_EXPORT(DotFloat32);
HWY_EXPORT(DotFloat64);

// Priority 1: Select Operations
HWY_EXPORT(SelectFloat32);
HWY_EXPORT(SelectFloat64);
HWY_EXPORT(SelectInt32);

// Priority 2: Extended Math Operations
HWY_EXPORT(Exp2Float32);
HWY_EXPORT(Exp2Float64);
HWY_EXPORT(Log2Float32);
HWY_EXPORT(Log2Float64);
HWY_EXPORT(Log10Float32);
HWY_EXPORT(Log10Float64);
HWY_EXPORT(SinhFloat32);
HWY_EXPORT(SinhFloat64);
HWY_EXPORT(CoshFloat32);
HWY_EXPORT(CoshFloat64);

// Priority 2: Inverse Trigonometric Operations
HWY_EXPORT(AsinFloat32);
HWY_EXPORT(AsinFloat64);
HWY_EXPORT(AcosFloat32);
HWY_EXPORT(AcosFloat64);
HWY_EXPORT(AtanFloat32);
HWY_EXPORT(AtanFloat64);
HWY_EXPORT(Atan2Float32);
HWY_EXPORT(Atan2Float64);

// Priority 2: Rounding Operations
HWY_EXPORT(RoundFloat32);
HWY_EXPORT(RoundFloat64);
HWY_EXPORT(FloorFloat32);
HWY_EXPORT(FloorFloat64);
HWY_EXPORT(CeilFloat32);
HWY_EXPORT(CeilFloat64);
HWY_EXPORT(TruncFloat32);
HWY_EXPORT(TruncFloat64);

// Priority 2: Bitwise Operations
HWY_EXPORT(BitwiseAndInt32);
HWY_EXPORT(BitwiseAndInt64);
HWY_EXPORT(BitwiseOrInt32);
HWY_EXPORT(BitwiseOrInt64);
HWY_EXPORT(BitwiseXorInt32);
HWY_EXPORT(BitwiseXorInt64);
HWY_EXPORT(BitwiseNotInt32);
HWY_EXPORT(BitwiseNotInt64);

// Priority 3: Special Value Checks
HWY_EXPORT(IsNaNFloat32);
HWY_EXPORT(IsNaNFloat64);
HWY_EXPORT(IsInfFloat32);
HWY_EXPORT(IsInfFloat64);
HWY_EXPORT(IsFiniteFloat32);
HWY_EXPORT(IsFiniteFloat64);

// Priority 3: Variable Shift Operations
HWY_EXPORT(ShiftLeftVarInt32);
HWY_EXPORT(ShiftLeftVarInt64);
HWY_EXPORT(ShiftLeftVarUint32);
HWY_EXPORT(ShiftRightVarInt32);
HWY_EXPORT(ShiftRightVarInt64);
HWY_EXPORT(ShiftRightVarUint32);

// Priority 3: Type Conversions
HWY_EXPORT(PromoteInt16ToInt32);
HWY_EXPORT(PromoteUint8ToInt32);
HWY_EXPORT(PromoteFloat32ToFloat64);
HWY_EXPORT(DemoteInt32ToInt16);
HWY_EXPORT(DemoteFloat64ToFloat32);
HWY_EXPORT(ConvertInt32ToFloat32);
HWY_EXPORT(ConvertFloat32ToInt32);

// Priority 3: Gather/Scatter
HWY_EXPORT(GatherFloat32);
HWY_EXPORT(GatherFloat64);
HWY_EXPORT(GatherInt32);
HWY_EXPORT(ScatterFloat32);
HWY_EXPORT(ScatterFloat64);
HWY_EXPORT(ScatterInt32);

// Priority 3: Horizontal Reductions
HWY_EXPORT(SumOfLanesFloat32);
// SumOfLanesFloat64, SumOfLanesInt32 use scalar fallbacks
HWY_EXPORT(PairwiseAddFloat32);
HWY_EXPORT(PairwiseAddFloat64);
HWY_EXPORT(PairwiseAddInt32);

// Priority 3: Saturation Operations
HWY_EXPORT(SaturatedAddInt16);
HWY_EXPORT(SaturatedAddUint8);
HWY_EXPORT(SaturatedSubInt16);
HWY_EXPORT(SaturatedSubUint8);

// Priority 3: Broadcast
HWY_EXPORT(BroadcastFloat32);
HWY_EXPORT(BroadcastFloat64);
HWY_EXPORT(BroadcastInt32);
HWY_EXPORT(BroadcastInt64);

// Priority 3: Compress/Expand
HWY_EXPORT(CompressFloat32);
HWY_EXPORT(CompressFloat64);
HWY_EXPORT(CompressInt32);
HWY_EXPORT(ExpandFloat32);
// ExpandFloat64, ExpandInt32 use scalar fallbacks

// Priority 3: Interleave
HWY_EXPORT(InterleaveFloat32);
HWY_EXPORT(InterleaveFloat64);
HWY_EXPORT(DeinterleaveFloat32);

// Priority 3: 8/16-bit Integer Operations
HWY_EXPORT(AddInt8);
HWY_EXPORT(SubInt8);
// MulInt8 uses scalar fallback
HWY_EXPORT(MinInt8);
HWY_EXPORT(MaxInt8);
HWY_EXPORT(AddInt16);
HWY_EXPORT(SubInt16);
HWY_EXPORT(MulInt16);
HWY_EXPORT(MinInt16);
HWY_EXPORT(MaxInt16);
HWY_EXPORT(AddUint8);
HWY_EXPORT(SubUint8);
// MulUint8 uses scalar fallback
HWY_EXPORT(MinUint8);
HWY_EXPORT(MaxUint8);
HWY_EXPORT(AddUint16);
HWY_EXPORT(SubUint16);
// MulUint16 uses scalar fallback
HWY_EXPORT(MinUint16);
HWY_EXPORT(MaxUint16);
HWY_EXPORT(AddUint32);
HWY_EXPORT(SubUint32);
HWY_EXPORT(MulUint32);
HWY_EXPORT(MinUint32);
HWY_EXPORT(MaxUint32);
HWY_EXPORT(AddUint64);
HWY_EXPORT(SubUint64);
// MulUint64, MinUint64, MaxUint64 use scalar fallback - no export needed

// Priority 3: Additional Math Operations
HWY_EXPORT(TanFloat32);
HWY_EXPORT(TanFloat64);
HWY_EXPORT(Expm1Float32);
HWY_EXPORT(Expm1Float64);
HWY_EXPORT(Log1pFloat32);
HWY_EXPORT(Log1pFloat64);
HWY_EXPORT(AbsDiffFloat32);
HWY_EXPORT(AbsDiffFloat64);
HWY_EXPORT(CopySignFloat32);
HWY_EXPORT(CopySignFloat64);
HWY_EXPORT(ApproxReciprocalFloat32);
HWY_EXPORT(ApproxReciprocalSqrtFloat32);
HWY_EXPORT(ApproxReciprocalSqrtFloat64);

// Priority 3: Bit Counting
HWY_EXPORT(PopCountInt32);
HWY_EXPORT(PopCountInt64);
HWY_EXPORT(PopCountUint32);
HWY_EXPORT(PopCountUint64);
HWY_EXPORT(LeadingZeroCountInt32);
HWY_EXPORT(LeadingZeroCountInt64);
HWY_EXPORT(LeadingZeroCountUint32);
HWY_EXPORT(LeadingZeroCountUint64);
HWY_EXPORT(TrailingZeroCountInt32);
HWY_EXPORT(TrailingZeroCountInt64);
HWY_EXPORT(TrailingZeroCountUint32);
HWY_EXPORT(TrailingZeroCountUint64);

// Priority 3: Averaging
HWY_EXPORT(AverageRoundUint8);
HWY_EXPORT(AverageRoundUint16);

// P0: Load/Store Operations
HWY_EXPORT(LoadFloat32);
HWY_EXPORT(LoadFloat64);
HWY_EXPORT(LoadInt32);
HWY_EXPORT(LoadInt64);
HWY_EXPORT(LoadUint8);
HWY_EXPORT(LoadUint16);
HWY_EXPORT(LoadUint32);
HWY_EXPORT(LoadUint64);
HWY_EXPORT(StoreFloat32);
HWY_EXPORT(StoreFloat64);
HWY_EXPORT(StoreInt32);
HWY_EXPORT(StoreInt64);
HWY_EXPORT(StoreUint8);
HWY_EXPORT(StoreUint16);
HWY_EXPORT(StoreUint32);
HWY_EXPORT(StoreUint64);

// P0: BitCast Operations
HWY_EXPORT(BitCastFloat32ToInt32Impl);
HWY_EXPORT(BitCastInt32ToFloat32Impl);
HWY_EXPORT(BitCastFloat64ToInt64Impl);
HWY_EXPORT(BitCastInt64ToFloat64Impl);
HWY_EXPORT(BitCastFloat32ToUint32Impl);
HWY_EXPORT(BitCastUint32ToFloat32Impl);
HWY_EXPORT(BitCastFloat64ToUint64Impl);
HWY_EXPORT(BitCastUint64ToFloat64Impl);

// P0: Mask Operations
HWY_EXPORT(CountTrueUint8Impl);
HWY_EXPORT(CountTrueUint32Impl);
HWY_EXPORT(CountTrueUint64Impl);
HWY_EXPORT(AllTrueUint8Impl);
HWY_EXPORT(AllTrueUint32Impl);
HWY_EXPORT(AllTrueUint64Impl);
HWY_EXPORT(AllFalseUint8Impl);
HWY_EXPORT(AllFalseUint32Impl);
HWY_EXPORT(AllFalseUint64Impl);
HWY_EXPORT(FindFirstTrueUint8Impl);
HWY_EXPORT(FindFirstTrueUint32Impl);
HWY_EXPORT(FindFirstTrueUint64Impl);
HWY_EXPORT(FindLastTrueUint8Impl);
HWY_EXPORT(FindLastTrueUint32Impl);
HWY_EXPORT(FindLastTrueUint64Impl);

// P1: Reverse Operations
HWY_EXPORT(ReverseFloat32);
HWY_EXPORT(ReverseFloat64);
HWY_EXPORT(ReverseInt32);
HWY_EXPORT(ReverseInt64);
HWY_EXPORT(ReverseUint8);
HWY_EXPORT(ReverseUint16);
HWY_EXPORT(ReverseUint32);
HWY_EXPORT(ReverseUint64);

// P1: Fill Operations
HWY_EXPORT(FillFloat32);
HWY_EXPORT(FillFloat64);
HWY_EXPORT(FillInt32);
HWY_EXPORT(FillInt64);
HWY_EXPORT(FillUint8);

// P1: Copy Operations
HWY_EXPORT(CopyFloat32);
HWY_EXPORT(CopyFloat64);
HWY_EXPORT(CopyInt32);
HWY_EXPORT(CopyInt64);
HWY_EXPORT(CopyUint8);

// P1: MinOfLanes/MaxOfLanes Operations
HWY_EXPORT(MinOfLanesFloat32);
HWY_EXPORT(MinOfLanesFloat64);
HWY_EXPORT(MinOfLanesInt32);
HWY_EXPORT(MinOfLanesInt64);
HWY_EXPORT(MinOfLanesUint32);
HWY_EXPORT(MaxOfLanesFloat32);
HWY_EXPORT(MaxOfLanesFloat64);
HWY_EXPORT(MaxOfLanesInt32);
HWY_EXPORT(MaxOfLanesInt64);
HWY_EXPORT(MaxOfLanesUint32);

// P1: Interleaved Load/Store Operations
HWY_EXPORT(LoadInterleaved2Float32);
HWY_EXPORT(LoadInterleaved2Int32);
HWY_EXPORT(LoadInterleaved2Uint8);
HWY_EXPORT(LoadInterleaved3Float32);
HWY_EXPORT(LoadInterleaved3Uint8);
HWY_EXPORT(LoadInterleaved4Float32);
HWY_EXPORT(LoadInterleaved4Uint8);
HWY_EXPORT(StoreInterleaved2Float32);
HWY_EXPORT(StoreInterleaved2Int32);
HWY_EXPORT(StoreInterleaved2Uint8);
HWY_EXPORT(StoreInterleaved3Float32);
HWY_EXPORT(StoreInterleaved3Uint8);
HWY_EXPORT(StoreInterleaved4Float32);
HWY_EXPORT(StoreInterleaved4Uint8);

// P1: Find Operations
HWY_EXPORT(FindFloat32);
HWY_EXPORT(FindInt32);
HWY_EXPORT(FindUint8);
HWY_EXPORT(FindGtFloat32);
HWY_EXPORT(FindGtInt32);
HWY_EXPORT(FindLtFloat32);
HWY_EXPORT(FindLtInt32);

// P1: Transform Operations
HWY_EXPORT(TransformAddFloat32);
HWY_EXPORT(TransformAddInt32);
HWY_EXPORT(TransformMulFloat32);
HWY_EXPORT(TransformMulInt32);

// P2: Hypot (sqrt(a^2 + b^2))
HWY_EXPORT(HypotFloat32);
HWY_EXPORT(HypotFloat64);

// P2: MatVec (Matrix-vector multiplication)
HWY_EXPORT(MatVecFloat32);
HWY_EXPORT(MatVecFloat64);

// P2: Pow (Power function)
HWY_EXPORT(PowFloat32);
HWY_EXPORT(PowFloat64);
HWY_EXPORT(PowScalarFloat32);
HWY_EXPORT(PowScalarFloat64);

// P0: Masked Arithmetic Operations
HWY_EXPORT(MaskedAddFloat32);
HWY_EXPORT(MaskedAddFloat64);
HWY_EXPORT(MaskedAddInt32);
HWY_EXPORT(MaskedAddInt64);
HWY_EXPORT(MaskedSubFloat32);
HWY_EXPORT(MaskedSubFloat64);
HWY_EXPORT(MaskedSubInt32);
HWY_EXPORT(MaskedSubInt64);
HWY_EXPORT(MaskedMulFloat32);
HWY_EXPORT(MaskedMulFloat64);
HWY_EXPORT(MaskedMulInt32);
HWY_EXPORT(MaskedMulInt64);
HWY_EXPORT(MaskedDivFloat32);
HWY_EXPORT(MaskedDivFloat64);
HWY_EXPORT(MaskedMinFloat32);
HWY_EXPORT(MaskedMinFloat64);
HWY_EXPORT(MaskedMinInt32);
HWY_EXPORT(MaskedMaxFloat32);
HWY_EXPORT(MaskedMaxFloat64);
HWY_EXPORT(MaskedMaxInt32);
HWY_EXPORT(MaskedAbsFloat32);
HWY_EXPORT(MaskedAbsFloat64);
HWY_EXPORT(MaskedAbsInt32);
HWY_EXPORT(MaskedNegFloat32);
HWY_EXPORT(MaskedNegFloat64);
HWY_EXPORT(MaskedNegInt32);

// P0: Widening Operations
HWY_EXPORT(SumsOf2Int16);
HWY_EXPORT(SumsOf2Uint16);
HWY_EXPORT(SumsOf4Int8);
HWY_EXPORT(SumsOf4Uint8);
HWY_EXPORT(MulEvenInt32);
HWY_EXPORT(MulEvenUint32);
HWY_EXPORT(MulOddInt32);
HWY_EXPORT(MulOddUint32);

// P0: Additional Comparison Operations
HWY_EXPORT(IsNegativeFloat32);
HWY_EXPORT(IsNegativeFloat64);
HWY_EXPORT(IsNegativeInt32);
HWY_EXPORT(IsEitherNaNFloat32);
HWY_EXPORT(IsEitherNaNFloat64);

// P1: Extended FMA Operations
HWY_EXPORT(NegMulSubFloat32);
HWY_EXPORT(NegMulSubFloat64);
HWY_EXPORT(MulAddSubFloat32);
HWY_EXPORT(MulAddSubFloat64);

// P1: CompressStore Operations
HWY_EXPORT(CompressStoreFloat32);
HWY_EXPORT(CompressStoreFloat64);
HWY_EXPORT(CompressStoreInt32);

// P2: Iota and FirstN Operations
HWY_EXPORT(IotaFloat32);
HWY_EXPORT(IotaFloat64);
HWY_EXPORT(IotaInt32);
HWY_EXPORT(IotaInt64);
HWY_EXPORT(IotaUint32);
HWY_EXPORT(IotaUint64);
HWY_EXPORT(FirstNImpl);

// P0: WidenMulAccumulate Operations
HWY_EXPORT(WidenMulAccumulateInt16);
HWY_EXPORT(WidenMulAccumulateUint16);

// P0: SumsOf8 Operation
HWY_EXPORT(SumsOf8Uint8);

// P0: TestBit Operation
HWY_EXPORT(TestBitInt32);
HWY_EXPORT(TestBitInt64);
HWY_EXPORT(TestBitUint32);
HWY_EXPORT(TestBitUint64);

// P1: MulSubAdd Operations
HWY_EXPORT(MulSubAddFloat32);
HWY_EXPORT(MulSubAddFloat64);

// P1: Reverse2, Reverse4, Reverse8 Operations
HWY_EXPORT(Reverse2Float32);
HWY_EXPORT(Reverse2Float64);
HWY_EXPORT(Reverse2Int32);
HWY_EXPORT(Reverse4Float32);
HWY_EXPORT(Reverse4Int32);
HWY_EXPORT(Reverse8Uint8);

// P1: DupEven, DupOdd Operations
HWY_EXPORT(DupEvenFloat32);
HWY_EXPORT(DupEvenInt32);
HWY_EXPORT(DupOddFloat32);
HWY_EXPORT(DupOddInt32);

// P1: InterleaveLower, InterleaveUpper Operations
HWY_EXPORT(InterleaveLowerFloat32);
HWY_EXPORT(InterleaveLowerInt32);
HWY_EXPORT(InterleaveUpperFloat32);
HWY_EXPORT(InterleaveUpperInt32);

// P1: Mask Logical Operations
HWY_EXPORT(MaskNotUint8);
HWY_EXPORT(MaskAndUint8);
HWY_EXPORT(MaskOrUint8);
HWY_EXPORT(MaskXorUint8);
HWY_EXPORT(MaskAndNotUint8);

// P2: AddSub Operations
HWY_EXPORT(AddSubFloat32);
HWY_EXPORT(AddSubFloat64);

// P2: MinMagnitude, MaxMagnitude Operations
HWY_EXPORT(MinMagnitudeFloat32);
HWY_EXPORT(MinMagnitudeFloat64);
HWY_EXPORT(MaxMagnitudeFloat32);
HWY_EXPORT(MaxMagnitudeFloat64);

// P2: MaskedLoad, MaskedStore, BlendedStore Operations
HWY_EXPORT(MaskedLoadFloat32);
HWY_EXPORT(MaskedLoadFloat64);
HWY_EXPORT(MaskedLoadInt32);
HWY_EXPORT(MaskedStoreFloat32);
HWY_EXPORT(MaskedStoreFloat64);
HWY_EXPORT(MaskedStoreInt32);
HWY_EXPORT(BlendedStoreFloat32);
HWY_EXPORT(BlendedStoreFloat64);
HWY_EXPORT(BlendedStoreInt32);

// P0: WidenMulPairwiseAdd Operations
HWY_EXPORT(WidenMulPairwiseAddInt16);
HWY_EXPORT(WidenMulPairwiseAddUint16);

// P1: BroadcastLane Operations
HWY_EXPORT(BroadcastLaneFloat32);
HWY_EXPORT(BroadcastLaneFloat64);
HWY_EXPORT(BroadcastLaneInt32);

// P1: Slide Operations
HWY_EXPORT(Slide1UpFloat32);
HWY_EXPORT(Slide1UpInt32);
HWY_EXPORT(Slide1DownFloat32);
HWY_EXPORT(Slide1DownInt32);

// P1: Concat Operations
HWY_EXPORT(ConcatLowerUpperFloat32);
HWY_EXPORT(ConcatLowerUpperInt32);
HWY_EXPORT(ConcatUpperLowerFloat32);
HWY_EXPORT(ConcatUpperLowerInt32);
HWY_EXPORT(ConcatEvenFloat32);
HWY_EXPORT(ConcatEvenInt32);
HWY_EXPORT(ConcatOddFloat32);
HWY_EXPORT(ConcatOddInt32);

// P1: Mask Utility Operations
HWY_EXPORT(FindKnownFirstTrueImpl);
HWY_EXPORT(FindKnownLastTrueImpl);
HWY_EXPORT(StoreMaskBitsImpl);
HWY_EXPORT(LoadMaskBitsImpl);

// P1: CompressBlendedStore and CompressNot Operations
HWY_EXPORT(CompressBlendedStoreFloat32);
HWY_EXPORT(CompressBlendedStoreInt32);
HWY_EXPORT(CompressNotFloat32);
HWY_EXPORT(CompressNotInt32);

// P1: InterleaveEven and InterleaveOdd Operations
HWY_EXPORT(InterleaveEvenFloat32);
HWY_EXPORT(InterleaveEvenInt32);
HWY_EXPORT(InterleaveOddFloat32);
HWY_EXPORT(InterleaveOddInt32);

// P1: Shuffle Operations
HWY_EXPORT(Shuffle0123Float32);
HWY_EXPORT(Shuffle0123Int32);
HWY_EXPORT(Shuffle2301Float32);
HWY_EXPORT(Shuffle2301Int32);
HWY_EXPORT(Shuffle1032Float32);
HWY_EXPORT(Shuffle1032Int32);
HWY_EXPORT(Shuffle01Float64);
HWY_EXPORT(Shuffle01Int64);
HWY_EXPORT(Shuffle10Float64);
HWY_EXPORT(Shuffle10Int64);

// P1: TableLookup Operations
HWY_EXPORT(TableLookupBytesUint8);
HWY_EXPORT(TableLookupLanesFloat32);
HWY_EXPORT(TableLookupLanesInt32);

// P1: Mask Set Operations
HWY_EXPORT(SetBeforeFirstImpl);
HWY_EXPORT(SetAtOrBeforeFirstImpl);
HWY_EXPORT(SetOnlyFirstImpl);
HWY_EXPORT(SetAtOrAfterFirstImpl);

// P1: Masked Reduction Operations
HWY_EXPORT(MaskedReduceSumFloat32);
HWY_EXPORT(MaskedReduceSumFloat64);
HWY_EXPORT(MaskedReduceSumInt32);
HWY_EXPORT(MaskedReduceMinFloat32);
HWY_EXPORT(MaskedReduceMinFloat64);
HWY_EXPORT(MaskedReduceMinInt32);
HWY_EXPORT(MaskedReduceMaxFloat32);
HWY_EXPORT(MaskedReduceMaxFloat64);
HWY_EXPORT(MaskedReduceMaxInt32);

// Remaining P1 Operations
HWY_EXPORT(TwoTablesLookupLanesFloat32);
HWY_EXPORT(TwoTablesLookupLanesInt32);
HWY_EXPORT(CompressBitsFloat32);
HWY_EXPORT(CompressBitsInt32);
HWY_EXPORT(CompressBitsStoreFloat32);
HWY_EXPORT(CompressBitsStoreInt32);
HWY_EXPORT(LoadExpandFloat32);
HWY_EXPORT(LoadExpandInt32);
HWY_EXPORT(PairwiseSubFloat32);
HWY_EXPORT(PairwiseSubInt32);
HWY_EXPORT(SumsOfAdjQuadAbsDiffUint8);

// P2: Math Functions
HWY_EXPORT(CbrtFloat32);
HWY_EXPORT(CbrtFloat64);
HWY_EXPORT(ErfFloat32);
HWY_EXPORT(ErfFloat64);
HWY_EXPORT(ErfcFloat32);
HWY_EXPORT(ErfcFloat64);

// P2: Generation Operations
HWY_EXPORT(IndicesFromVecImpl);
HWY_EXPORT(IndicesFromNotVecImpl);

// P2: Type Conversions
HWY_EXPORT(PromoteLowerToFloat64);
HWY_EXPORT(PromoteLowerToInt32);
HWY_EXPORT(PromoteLowerToInt64);
HWY_EXPORT(PromoteUpperToFloat64);
HWY_EXPORT(PromoteUpperToInt32);
HWY_EXPORT(PromoteEvenToFloat64);
HWY_EXPORT(PromoteEvenToInt32);
HWY_EXPORT(PromoteOddToFloat64);
HWY_EXPORT(PromoteOddToInt32);

// P2: Additional Arithmetic
HWY_EXPORT(ModInt32);
HWY_EXPORT(ModInt64);
HWY_EXPORT(SaturatedNegInt8);
HWY_EXPORT(SaturatedNegInt16);
HWY_EXPORT(SaturatedAbsInt8);
HWY_EXPORT(SaturatedAbsInt16);

// P2: Bitwise Operations
HWY_EXPORT(Xor3Int32);
HWY_EXPORT(Xor3Int64);
HWY_EXPORT(Or3Int32);
HWY_EXPORT(Or3Int64);
HWY_EXPORT(OrAndInt32);
HWY_EXPORT(OrAndInt64);
HWY_EXPORT(ReverseBitsUint32);
HWY_EXPORT(ReverseBitsUint64);
HWY_EXPORT(HighestSetBitIndexUint32);
HWY_EXPORT(HighestSetBitIndexUint64);

// P2: Memory Operations
HWY_EXPORT(LoadDup128Float32);
HWY_EXPORT(LoadDup128Float64);
HWY_EXPORT(GatherOffsetFloat32);
HWY_EXPORT(GatherOffsetInt32);
HWY_EXPORT(ScatterOffsetFloat32);
HWY_EXPORT(ScatterOffsetInt32);
HWY_EXPORT(MaskedGatherIndexFloat32);
HWY_EXPORT(MaskedGatherIndexInt32);
HWY_EXPORT(MaskedScatterIndexFloat32);
HWY_EXPORT(MaskedScatterIndexInt32);
HWY_EXPORT(SafeFillNFloat32);
HWY_EXPORT(SafeFillNInt32);
HWY_EXPORT(SafeCopyNFloat32);
HWY_EXPORT(SafeCopyNInt32);

// P2: Special Operations
HWY_EXPORT(MulByPow2Float32);
HWY_EXPORT(MulByPow2Float64);
HWY_EXPORT(GetExponentFloat32);
HWY_EXPORT(GetExponentFloat64);
HWY_EXPORT(SignBitFloat32);
HWY_EXPORT(SignBitFloat64);
HWY_EXPORT(NaNFloat32);
HWY_EXPORT(NaNFloat64);
HWY_EXPORT(InfFloat32);
HWY_EXPORT(InfFloat64);

// P3: Complex Number Operations
HWY_EXPORT(ComplexConjFloat32);
HWY_EXPORT(ComplexConjFloat64);
HWY_EXPORT(MulComplexFloat32);
HWY_EXPORT(MulComplexFloat64);
HWY_EXPORT(MulComplexAddFloat32);

// P3: Saturation Operations
HWY_EXPORT(SaturatedAddInt8);
HWY_EXPORT(SaturatedSubInt8);
HWY_EXPORT(SaturatedAddUint16);
HWY_EXPORT(SaturatedSubUint16);

// P3: Block Operations
HWY_EXPORT(SlideUpBlocksFloat32);
HWY_EXPORT(SlideDownBlocksFloat32);
HWY_EXPORT(CombineShiftRightLanesFloat32);

// P3: Additional Masked Operations
HWY_EXPORT(MaskedSqrtFloat32);
HWY_EXPORT(MaskedSqrtFloat64);
HWY_EXPORT(ZeroIfNegativeFloat32);
HWY_EXPORT(ZeroIfNegativeFloat64);

// P2: Additional Type Conversion Operations
HWY_EXPORT(ReorderDemote2ToInt32Int16);
HWY_EXPORT(ReorderDemote2ToUint32Uint16);
HWY_EXPORT(OrderedTruncate2ToInt32Int16);
HWY_EXPORT(OrderedTruncate2ToUint32Uint16);
HWY_EXPORT(ConvertInRangeToFloat32Int32);
HWY_EXPORT(ConvertInRangeToFloat64Int64);
HWY_EXPORT(ResizeBitCastFloat32Uint32);
HWY_EXPORT(ResizeBitCastUint32Float32);
HWY_EXPORT(ResizeBitCastFloat64Uint64);
HWY_EXPORT(ResizeBitCastUint64Float64);

// P2: Additional Special Operations
HWY_EXPORT(MulByFloorPow2Float32);
HWY_EXPORT(MulByFloorPow2Float64);
HWY_EXPORT(GetBiasedExponentFloat32);
HWY_EXPORT(GetBiasedExponentFloat64);
HWY_EXPORT(MulFixedPoint15Int16);
HWY_EXPORT(MulRoundInt16);
HWY_EXPORT(MulRoundInt32);
HWY_EXPORT(RoundingShiftRightInt16);
HWY_EXPORT(RoundingShiftRightInt32);

// P3: Additional Complex Number Operations
HWY_EXPORT(MulComplexConjFloat32);
HWY_EXPORT(MulComplexConjFloat64);
HWY_EXPORT(MulComplexConjAddFloat32);
HWY_EXPORT(MulComplexConjAddFloat64);

// P3: Per-Lane Block Shuffle
HWY_EXPORT(Per4LaneBlockShuffleFloat32);
HWY_EXPORT(Per4LaneBlockShuffleInt32);

// P3: Additional Masked Operations
HWY_EXPORT(MaskedReciprocalFloat32);
HWY_EXPORT(MaskedReciprocalFloat64);
HWY_EXPORT(MaskedShiftLeftInt32);
HWY_EXPORT(MaskedShiftLeftInt64);
HWY_EXPORT(MaskedShiftRightInt32);
HWY_EXPORT(MaskedShiftRightInt64);
HWY_EXPORT(MaskedSatAddInt8);
HWY_EXPORT(MaskedSatAddInt16);
HWY_EXPORT(MaskedSatSubInt8);
HWY_EXPORT(MaskedSatSubInt16);

// P3: Masked Comparison Operations
HWY_EXPORT(MaskedEqFloat32);
HWY_EXPORT(MaskedEqInt32);
HWY_EXPORT(MaskedNeFloat32);
HWY_EXPORT(MaskedNeInt32);
HWY_EXPORT(MaskedLtFloat32);
HWY_EXPORT(MaskedLtInt32);
HWY_EXPORT(MaskedLeFloat32);
HWY_EXPORT(MaskedLeInt32);
HWY_EXPORT(MaskedGtFloat32);
HWY_EXPORT(MaskedGtInt32);
HWY_EXPORT(MaskedGeFloat32);
HWY_EXPORT(MaskedGeInt32);

// P3.2: Cryptographic Operations
HWY_EXPORT(AESRoundImpl);
HWY_EXPORT(AESLastRoundImpl);
HWY_EXPORT(AESRoundInvImpl);
HWY_EXPORT(AESLastRoundInvImpl);
HWY_EXPORT(AESInvMixColumnsImpl);
HWY_EXPORT(AESKeyGenAssistImpl);
HWY_EXPORT(CLMulLowerImpl);
HWY_EXPORT(CLMulUpperImpl);

// P3.3: Random Number Generation
HWY_EXPORT(RandomStateInitImpl);
HWY_EXPORT(Random32Impl);
HWY_EXPORT(Random64Impl);
HWY_EXPORT(RandomFloatImpl);

// P3.4: Bit Packing
HWY_EXPORT(PackBitsImpl);
HWY_EXPORT(UnpackBitsImpl);

// P3.6: Algorithm Operations
HWY_EXPORT(FindIfGreaterThanFloat32);
HWY_EXPORT(FindIfGreaterThanInt32);
HWY_EXPORT(GenerateFloat32);
HWY_EXPORT(GenerateInt32);
HWY_EXPORT(ReplaceFloat32);
HWY_EXPORT(ReplaceInt32);
HWY_EXPORT(ReplaceIfGreaterThanFloat32);
HWY_EXPORT(ReplaceIfGreaterThanInt32);

// P3.5: Image Processing Operations
HWY_EXPORT(ImageFillImpl);
HWY_EXPORT(ImageCopyImpl);
HWY_EXPORT(ImageAddImpl);
HWY_EXPORT(ImageSubImpl);
HWY_EXPORT(ImageMulImpl);
HWY_EXPORT(ImageScaleImpl);
HWY_EXPORT(ImageClampImpl);
HWY_EXPORT(Convolve3x3Impl);
HWY_EXPORT(BoxBlur3x3Impl);
HWY_EXPORT(GaussianBlur3x3Impl);
HWY_EXPORT(SobelEdgeImpl);
HWY_EXPORT(SharpenImpl);
HWY_EXPORT(ThresholdImpl);
HWY_EXPORT(GrayscaleImpl);
HWY_EXPORT(Downsample2xImpl);
HWY_EXPORT(Upsample2xImpl);

// Gap Operations: Additional SIMD Operations
HWY_EXPORT(OddEvenFloat32);
HWY_EXPORT(OddEvenInt32);
HWY_EXPORT(MaskedMulAddFloat32);
HWY_EXPORT(MaskedNegMulAddFloat32);
HWY_EXPORT(InterleaveWholeLowerFloat32);
HWY_EXPORT(InterleaveWholeUpperFloat32);
HWY_EXPORT(IfNegativeThenElseZeroFloat32);
HWY_EXPORT(IfNegativeThenZeroElseFloat32);
HWY_EXPORT(BitwiseIfThenElseInt32);
HWY_EXPORT(MaskFalseImpl);
HWY_EXPORT(SetMaskImpl);
HWY_EXPORT(CeilIntFloat32);
HWY_EXPORT(FloorIntFloat32);
HWY_EXPORT(TruncateStoreInt32ToInt16);
HWY_EXPORT(MaskedModOrInt32);

// Final gap operations
HWY_EXPORT(Per2LaneBlockShuffleFloat32);
HWY_EXPORT(MaskedIsNaNFloat32);
HWY_EXPORT(IfNegativeThenNegOrUndefIfZeroFloat32);
HWY_EXPORT(MaskedSetOrFloat32);
HWY_EXPORT(MaskedMulFixedPoint15Int16);
HWY_EXPORT(MaskedWidenMulPairwiseAddInt16ToInt32);
HWY_EXPORT(MaskedAbsOrFloat32);
HWY_EXPORT(InsertIntoUpperFloat32);
HWY_EXPORT(MaskedGatherIndexOrFloat32);
HWY_EXPORT(SumsOf8AbsDiffUint8);
HWY_EXPORT(CombineMasksImpl);
HWY_EXPORT(LowerHalfOfMaskImpl);
HWY_EXPORT(UpperHalfOfMaskImpl);
HWY_EXPORT(PromoteMaskToUint8ToUint16);
HWY_EXPORT(DemoteMaskToUint16ToUint8);
HWY_EXPORT(ZeroExtendResizeBitCastUint8ToUint32);
HWY_EXPORT(F64ToF16Impl);
HWY_EXPORT(F64ToBF16Impl);
HWY_EXPORT(MatVecMulFloat32);
HWY_EXPORT(MatMulFloat32);

}  // namespace hwy

// =============================================================================
// Public API - Dynamic Dispatch Wrappers in bud::simd namespace
// =============================================================================
namespace bud {
namespace simd {

// Import dispatch tables from hwy namespace
using hwy::AbsFloat32HighwayDispatchTable;
using hwy::AbsFloat64HighwayDispatchTable;
using hwy::AbsInt32HighwayDispatchTable;
using hwy::AbsInt64HighwayDispatchTable;
using hwy::AcosFloat32HighwayDispatchTable;
using hwy::AcosFloat64HighwayDispatchTable;
using hwy::AddFloat32HighwayDispatchTable;
using hwy::AddFloat64HighwayDispatchTable;
using hwy::AddInt32HighwayDispatchTable;
using hwy::AddInt64HighwayDispatchTable;
using hwy::AsinFloat32HighwayDispatchTable;
using hwy::AsinFloat64HighwayDispatchTable;
using hwy::Atan2Float32HighwayDispatchTable;
using hwy::Atan2Float64HighwayDispatchTable;
using hwy::AtanFloat32HighwayDispatchTable;
using hwy::AtanFloat64HighwayDispatchTable;
using hwy::BitwiseAndInt32HighwayDispatchTable;
using hwy::BitwiseAndInt64HighwayDispatchTable;
using hwy::BitwiseNotInt32HighwayDispatchTable;
using hwy::BitwiseNotInt64HighwayDispatchTable;
using hwy::BitwiseOrInt32HighwayDispatchTable;
using hwy::BitwiseOrInt64HighwayDispatchTable;
using hwy::BitwiseXorInt32HighwayDispatchTable;
using hwy::BitwiseXorInt64HighwayDispatchTable;
using hwy::CeilFloat32HighwayDispatchTable;
using hwy::CeilFloat64HighwayDispatchTable;
using hwy::ClampFloat32HighwayDispatchTable;
using hwy::ClampFloat64HighwayDispatchTable;
using hwy::ClampInt32HighwayDispatchTable;
using hwy::ClampInt64HighwayDispatchTable;
using hwy::CosFloat32HighwayDispatchTable;
using hwy::CosFloat64HighwayDispatchTable;
using hwy::CoshFloat32HighwayDispatchTable;
using hwy::CoshFloat64HighwayDispatchTable;
using hwy::DivFloat32HighwayDispatchTable;
using hwy::DivFloat64HighwayDispatchTable;
using hwy::DotFloat32HighwayDispatchTable;
using hwy::DotFloat64HighwayDispatchTable;
using hwy::EqFloat32HighwayDispatchTable;
using hwy::EqFloat64HighwayDispatchTable;
using hwy::EqInt32HighwayDispatchTable;
using hwy::Exp2Float32HighwayDispatchTable;
using hwy::Exp2Float64HighwayDispatchTable;
using hwy::ExpFloat32HighwayDispatchTable;
using hwy::ExpFloat64HighwayDispatchTable;
using hwy::FloorFloat32HighwayDispatchTable;
using hwy::FloorFloat64HighwayDispatchTable;
using hwy::GeFloat32HighwayDispatchTable;
using hwy::GeFloat64HighwayDispatchTable;
using hwy::GeInt32HighwayDispatchTable;
using hwy::GtFloat32HighwayDispatchTable;
using hwy::GtFloat64HighwayDispatchTable;
using hwy::GtInt32HighwayDispatchTable;
using hwy::LeFloat32HighwayDispatchTable;
using hwy::LeFloat64HighwayDispatchTable;
using hwy::LeInt32HighwayDispatchTable;
using hwy::Log10Float32HighwayDispatchTable;
using hwy::Log10Float64HighwayDispatchTable;
using hwy::Log2Float32HighwayDispatchTable;
using hwy::Log2Float64HighwayDispatchTable;
using hwy::LogFloat32HighwayDispatchTable;
using hwy::LogFloat64HighwayDispatchTable;
using hwy::LtFloat32HighwayDispatchTable;
using hwy::LtFloat64HighwayDispatchTable;
using hwy::LtInt32HighwayDispatchTable;
using hwy::MaxFloat32HighwayDispatchTable;
using hwy::MaxFloat64HighwayDispatchTable;
using hwy::MaxInt32HighwayDispatchTable;
using hwy::MaxInt64HighwayDispatchTable;
using hwy::MinFloat32HighwayDispatchTable;
using hwy::MinFloat64HighwayDispatchTable;
using hwy::MinInt32HighwayDispatchTable;
using hwy::MinInt64HighwayDispatchTable;
using hwy::MulAddFloat32HighwayDispatchTable;
using hwy::MulAddFloat64HighwayDispatchTable;
using hwy::MulFloat32HighwayDispatchTable;
using hwy::MulFloat64HighwayDispatchTable;
using hwy::MulInt32HighwayDispatchTable;
using hwy::MulInt64HighwayDispatchTable;
using hwy::MulSubFloat32HighwayDispatchTable;
using hwy::MulSubFloat64HighwayDispatchTable;
using hwy::NeFloat32HighwayDispatchTable;
using hwy::NeFloat64HighwayDispatchTable;
using hwy::NegFloat32HighwayDispatchTable;
using hwy::NegFloat64HighwayDispatchTable;
using hwy::NegInt32HighwayDispatchTable;
using hwy::NegInt64HighwayDispatchTable;
using hwy::NegMulAddFloat32HighwayDispatchTable;
using hwy::NegMulAddFloat64HighwayDispatchTable;
using hwy::NeInt32HighwayDispatchTable;
using hwy::ReduceMaxFloat32HighwayDispatchTable;
using hwy::ReduceMaxFloat64HighwayDispatchTable;
using hwy::ReduceMaxInt32HighwayDispatchTable;
using hwy::ReduceMaxInt64HighwayDispatchTable;
using hwy::ReduceMinFloat32HighwayDispatchTable;
using hwy::ReduceMinFloat64HighwayDispatchTable;
using hwy::ReduceMinInt32HighwayDispatchTable;
using hwy::ReduceMinInt64HighwayDispatchTable;
using hwy::ReduceSumFloat32HighwayDispatchTable;
using hwy::ReduceSumFloat64HighwayDispatchTable;
using hwy::ReduceSumInt32HighwayDispatchTable;
using hwy::ReduceSumInt64HighwayDispatchTable;
using hwy::RoundFloat32HighwayDispatchTable;
using hwy::RoundFloat64HighwayDispatchTable;
using hwy::RsqrtFloat32HighwayDispatchTable;
using hwy::RsqrtFloat64HighwayDispatchTable;
using hwy::SelectFloat32HighwayDispatchTable;
using hwy::SelectFloat64HighwayDispatchTable;
using hwy::SelectInt32HighwayDispatchTable;
using hwy::SinFloat32HighwayDispatchTable;
using hwy::SinFloat64HighwayDispatchTable;
using hwy::SinhFloat32HighwayDispatchTable;
using hwy::SinhFloat64HighwayDispatchTable;
using hwy::SqrtFloat32HighwayDispatchTable;
using hwy::SqrtFloat64HighwayDispatchTable;
using hwy::SubFloat32HighwayDispatchTable;
using hwy::SubFloat64HighwayDispatchTable;
using hwy::SubInt32HighwayDispatchTable;
using hwy::SubInt64HighwayDispatchTable;
using hwy::TanhFloat32HighwayDispatchTable;
using hwy::TanhFloat64HighwayDispatchTable;
using hwy::TruncFloat32HighwayDispatchTable;
using hwy::TruncFloat64HighwayDispatchTable;

// Priority 3: Special Value Checks
using hwy::IsFiniteFloat32HighwayDispatchTable;
using hwy::IsFiniteFloat64HighwayDispatchTable;
using hwy::IsInfFloat32HighwayDispatchTable;
using hwy::IsInfFloat64HighwayDispatchTable;
using hwy::IsNaNFloat32HighwayDispatchTable;
using hwy::IsNaNFloat64HighwayDispatchTable;

// Priority 3: Variable Shift Operations
using hwy::ShiftLeftVarInt32HighwayDispatchTable;
using hwy::ShiftLeftVarInt64HighwayDispatchTable;
using hwy::ShiftLeftVarUint32HighwayDispatchTable;
using hwy::ShiftRightVarInt32HighwayDispatchTable;
using hwy::ShiftRightVarInt64HighwayDispatchTable;
using hwy::ShiftRightVarUint32HighwayDispatchTable;

// Priority 3: Type Conversions
using hwy::ConvertFloat32ToInt32HighwayDispatchTable;
using hwy::ConvertInt32ToFloat32HighwayDispatchTable;
using hwy::DemoteFloat64ToFloat32HighwayDispatchTable;
using hwy::DemoteInt32ToInt16HighwayDispatchTable;
using hwy::PromoteFloat32ToFloat64HighwayDispatchTable;
using hwy::PromoteInt16ToInt32HighwayDispatchTable;
using hwy::PromoteUint8ToInt32HighwayDispatchTable;

// Priority 3: Gather/Scatter
using hwy::GatherFloat32HighwayDispatchTable;
using hwy::GatherFloat64HighwayDispatchTable;
using hwy::GatherInt32HighwayDispatchTable;
using hwy::ScatterFloat32HighwayDispatchTable;
using hwy::ScatterFloat64HighwayDispatchTable;
using hwy::ScatterInt32HighwayDispatchTable;

// Priority 3: Horizontal Reductions
using hwy::SumOfLanesFloat32HighwayDispatchTable;
// SumOfLanesFloat64, SumOfLanesInt32 use scalar fallbacks
using hwy::PairwiseAddFloat32HighwayDispatchTable;
using hwy::PairwiseAddFloat64HighwayDispatchTable;
using hwy::PairwiseAddInt32HighwayDispatchTable;

// Priority 3: Saturation Operations
using hwy::SaturatedAddInt16HighwayDispatchTable;
using hwy::SaturatedAddUint8HighwayDispatchTable;
using hwy::SaturatedSubInt16HighwayDispatchTable;
using hwy::SaturatedSubUint8HighwayDispatchTable;

// Priority 3: Broadcast
using hwy::BroadcastFloat32HighwayDispatchTable;
using hwy::BroadcastFloat64HighwayDispatchTable;
using hwy::BroadcastInt32HighwayDispatchTable;
using hwy::BroadcastInt64HighwayDispatchTable;

// Priority 3: Compress/Expand
using hwy::CompressFloat32HighwayDispatchTable;
using hwy::CompressFloat64HighwayDispatchTable;
using hwy::CompressInt32HighwayDispatchTable;
using hwy::ExpandFloat32HighwayDispatchTable;
// ExpandFloat64, ExpandInt32 use scalar fallbacks

// Priority 3: Interleave
using hwy::DeinterleaveFloat32HighwayDispatchTable;
using hwy::InterleaveFloat32HighwayDispatchTable;
using hwy::InterleaveFloat64HighwayDispatchTable;

// Priority 3: 8/16-bit Integer Operations
using hwy::AddInt8HighwayDispatchTable;
using hwy::SubInt8HighwayDispatchTable;
// MulInt8 uses scalar fallback
using hwy::AddInt16HighwayDispatchTable;
using hwy::AddUint8HighwayDispatchTable;
using hwy::MaxInt16HighwayDispatchTable;
using hwy::MaxInt8HighwayDispatchTable;
using hwy::MinInt16HighwayDispatchTable;
using hwy::MinInt8HighwayDispatchTable;
using hwy::MulInt16HighwayDispatchTable;
using hwy::SubInt16HighwayDispatchTable;
using hwy::SubUint8HighwayDispatchTable;
// MulUint8 uses scalar fallback
using hwy::AddUint16HighwayDispatchTable;
using hwy::MaxUint8HighwayDispatchTable;
using hwy::MinUint8HighwayDispatchTable;
using hwy::SubUint16HighwayDispatchTable;
// MulUint16 uses scalar fallback
using hwy::AddUint32HighwayDispatchTable;
using hwy::AddUint64HighwayDispatchTable;
using hwy::MaxUint16HighwayDispatchTable;
using hwy::MaxUint32HighwayDispatchTable;
using hwy::MinUint16HighwayDispatchTable;
using hwy::MinUint32HighwayDispatchTable;
using hwy::MulUint32HighwayDispatchTable;
using hwy::SubUint32HighwayDispatchTable;
using hwy::SubUint64HighwayDispatchTable;
// MulUint64, MinUint64, MaxUint64 use scalar fallback

// Priority 3: Additional Math Operations
using hwy::AbsDiffFloat32HighwayDispatchTable;
using hwy::AbsDiffFloat64HighwayDispatchTable;
using hwy::ApproxReciprocalFloat32HighwayDispatchTable;
using hwy::ApproxReciprocalSqrtFloat32HighwayDispatchTable;
using hwy::ApproxReciprocalSqrtFloat64HighwayDispatchTable;
using hwy::CopySignFloat32HighwayDispatchTable;
using hwy::CopySignFloat64HighwayDispatchTable;
using hwy::Expm1Float32HighwayDispatchTable;
using hwy::Expm1Float64HighwayDispatchTable;
using hwy::Log1pFloat32HighwayDispatchTable;
using hwy::Log1pFloat64HighwayDispatchTable;
using hwy::TanFloat32HighwayDispatchTable;
using hwy::TanFloat64HighwayDispatchTable;

// Priority 3: Bit Counting
using hwy::LeadingZeroCountInt32HighwayDispatchTable;
using hwy::LeadingZeroCountInt64HighwayDispatchTable;
using hwy::LeadingZeroCountUint32HighwayDispatchTable;
using hwy::LeadingZeroCountUint64HighwayDispatchTable;
using hwy::PopCountInt32HighwayDispatchTable;
using hwy::PopCountInt64HighwayDispatchTable;
using hwy::PopCountUint32HighwayDispatchTable;
using hwy::PopCountUint64HighwayDispatchTable;
using hwy::TrailingZeroCountInt32HighwayDispatchTable;
using hwy::TrailingZeroCountInt64HighwayDispatchTable;
using hwy::TrailingZeroCountUint32HighwayDispatchTable;
using hwy::TrailingZeroCountUint64HighwayDispatchTable;

// Priority 3: Averaging
using hwy::AverageRoundUint16HighwayDispatchTable;
using hwy::AverageRoundUint8HighwayDispatchTable;

// P0: Load/Store Operations
using hwy::LoadFloat32HighwayDispatchTable;
using hwy::LoadFloat64HighwayDispatchTable;
using hwy::LoadInt32HighwayDispatchTable;
using hwy::LoadInt64HighwayDispatchTable;
using hwy::LoadUint16HighwayDispatchTable;
using hwy::LoadUint32HighwayDispatchTable;
using hwy::LoadUint64HighwayDispatchTable;
using hwy::LoadUint8HighwayDispatchTable;
using hwy::StoreFloat32HighwayDispatchTable;
using hwy::StoreFloat64HighwayDispatchTable;
using hwy::StoreInt32HighwayDispatchTable;
using hwy::StoreInt64HighwayDispatchTable;
using hwy::StoreUint16HighwayDispatchTable;
using hwy::StoreUint32HighwayDispatchTable;
using hwy::StoreUint64HighwayDispatchTable;
using hwy::StoreUint8HighwayDispatchTable;

// P0: BitCast Operations
using hwy::BitCastFloat32ToInt32ImplHighwayDispatchTable;
using hwy::BitCastFloat32ToUint32ImplHighwayDispatchTable;
using hwy::BitCastFloat64ToInt64ImplHighwayDispatchTable;
using hwy::BitCastFloat64ToUint64ImplHighwayDispatchTable;
using hwy::BitCastInt32ToFloat32ImplHighwayDispatchTable;
using hwy::BitCastInt64ToFloat64ImplHighwayDispatchTable;
using hwy::BitCastUint32ToFloat32ImplHighwayDispatchTable;
using hwy::BitCastUint64ToFloat64ImplHighwayDispatchTable;

// P0: Mask Operations
using hwy::AllFalseUint32ImplHighwayDispatchTable;
using hwy::AllFalseUint64ImplHighwayDispatchTable;
using hwy::AllFalseUint8ImplHighwayDispatchTable;
using hwy::AllTrueUint32ImplHighwayDispatchTable;
using hwy::AllTrueUint64ImplHighwayDispatchTable;
using hwy::AllTrueUint8ImplHighwayDispatchTable;
using hwy::CountTrueUint32ImplHighwayDispatchTable;
using hwy::CountTrueUint64ImplHighwayDispatchTable;
using hwy::CountTrueUint8ImplHighwayDispatchTable;
using hwy::FindFirstTrueUint32ImplHighwayDispatchTable;
using hwy::FindFirstTrueUint64ImplHighwayDispatchTable;
using hwy::FindFirstTrueUint8ImplHighwayDispatchTable;
using hwy::FindLastTrueUint32ImplHighwayDispatchTable;
using hwy::FindLastTrueUint64ImplHighwayDispatchTable;
using hwy::FindLastTrueUint8ImplHighwayDispatchTable;

// P1: Reverse Operations
using hwy::ReverseFloat32HighwayDispatchTable;
using hwy::ReverseFloat64HighwayDispatchTable;
using hwy::ReverseInt32HighwayDispatchTable;
using hwy::ReverseInt64HighwayDispatchTable;
using hwy::ReverseUint16HighwayDispatchTable;
using hwy::ReverseUint32HighwayDispatchTable;
using hwy::ReverseUint64HighwayDispatchTable;
using hwy::ReverseUint8HighwayDispatchTable;

// P1: Fill Operations
using hwy::FillFloat32HighwayDispatchTable;
using hwy::FillFloat64HighwayDispatchTable;
using hwy::FillInt32HighwayDispatchTable;
using hwy::FillInt64HighwayDispatchTable;
using hwy::FillUint8HighwayDispatchTable;

// P1: Copy Operations
using hwy::CopyFloat32HighwayDispatchTable;
using hwy::CopyFloat64HighwayDispatchTable;
using hwy::CopyInt32HighwayDispatchTable;
using hwy::CopyInt64HighwayDispatchTable;
using hwy::CopyUint8HighwayDispatchTable;

// P1: MinOfLanes/MaxOfLanes Operations
using hwy::MaxOfLanesFloat32HighwayDispatchTable;
using hwy::MaxOfLanesFloat64HighwayDispatchTable;
using hwy::MaxOfLanesInt32HighwayDispatchTable;
using hwy::MaxOfLanesInt64HighwayDispatchTable;
using hwy::MaxOfLanesUint32HighwayDispatchTable;
using hwy::MinOfLanesFloat32HighwayDispatchTable;
using hwy::MinOfLanesFloat64HighwayDispatchTable;
using hwy::MinOfLanesInt32HighwayDispatchTable;
using hwy::MinOfLanesInt64HighwayDispatchTable;
using hwy::MinOfLanesUint32HighwayDispatchTable;

// P1: Interleaved Load/Store Operations
using hwy::LoadInterleaved2Float32HighwayDispatchTable;
using hwy::LoadInterleaved2Int32HighwayDispatchTable;
using hwy::LoadInterleaved2Uint8HighwayDispatchTable;
using hwy::LoadInterleaved3Float32HighwayDispatchTable;
using hwy::LoadInterleaved3Uint8HighwayDispatchTable;
using hwy::LoadInterleaved4Float32HighwayDispatchTable;
using hwy::LoadInterleaved4Uint8HighwayDispatchTable;
using hwy::StoreInterleaved2Float32HighwayDispatchTable;
using hwy::StoreInterleaved2Int32HighwayDispatchTable;
using hwy::StoreInterleaved2Uint8HighwayDispatchTable;
using hwy::StoreInterleaved3Float32HighwayDispatchTable;
using hwy::StoreInterleaved3Uint8HighwayDispatchTable;
using hwy::StoreInterleaved4Float32HighwayDispatchTable;
using hwy::StoreInterleaved4Uint8HighwayDispatchTable;

// P1: Find Operations
using hwy::FindFloat32HighwayDispatchTable;
using hwy::FindGtFloat32HighwayDispatchTable;
using hwy::FindGtInt32HighwayDispatchTable;
using hwy::FindInt32HighwayDispatchTable;
using hwy::FindLtFloat32HighwayDispatchTable;
using hwy::FindLtInt32HighwayDispatchTable;
using hwy::FindUint8HighwayDispatchTable;

// P1: Transform Operations
using hwy::TransformAddFloat32HighwayDispatchTable;
using hwy::TransformAddInt32HighwayDispatchTable;
using hwy::TransformMulFloat32HighwayDispatchTable;
using hwy::TransformMulInt32HighwayDispatchTable;

// P2: Hypot
using hwy::HypotFloat32HighwayDispatchTable;
using hwy::HypotFloat64HighwayDispatchTable;

// P2: MatVec
using hwy::MatVecFloat32HighwayDispatchTable;
using hwy::MatVecFloat64HighwayDispatchTable;

// P2: Pow
using hwy::PowFloat32HighwayDispatchTable;
using hwy::PowFloat64HighwayDispatchTable;
using hwy::PowScalarFloat32HighwayDispatchTable;
using hwy::PowScalarFloat64HighwayDispatchTable;

// P0: Masked Arithmetic Operations
using hwy::MaskedAbsFloat32HighwayDispatchTable;
using hwy::MaskedAbsFloat64HighwayDispatchTable;
using hwy::MaskedAbsInt32HighwayDispatchTable;
using hwy::MaskedAddFloat32HighwayDispatchTable;
using hwy::MaskedAddFloat64HighwayDispatchTable;
using hwy::MaskedAddInt32HighwayDispatchTable;
using hwy::MaskedAddInt64HighwayDispatchTable;
using hwy::MaskedDivFloat32HighwayDispatchTable;
using hwy::MaskedDivFloat64HighwayDispatchTable;
using hwy::MaskedMaxFloat32HighwayDispatchTable;
using hwy::MaskedMaxFloat64HighwayDispatchTable;
using hwy::MaskedMaxInt32HighwayDispatchTable;
using hwy::MaskedMinFloat32HighwayDispatchTable;
using hwy::MaskedMinFloat64HighwayDispatchTable;
using hwy::MaskedMinInt32HighwayDispatchTable;
using hwy::MaskedMulFloat32HighwayDispatchTable;
using hwy::MaskedMulFloat64HighwayDispatchTable;
using hwy::MaskedMulInt32HighwayDispatchTable;
using hwy::MaskedMulInt64HighwayDispatchTable;
using hwy::MaskedNegFloat32HighwayDispatchTable;
using hwy::MaskedNegFloat64HighwayDispatchTable;
using hwy::MaskedNegInt32HighwayDispatchTable;
using hwy::MaskedSubFloat32HighwayDispatchTable;
using hwy::MaskedSubFloat64HighwayDispatchTable;
using hwy::MaskedSubInt32HighwayDispatchTable;
using hwy::MaskedSubInt64HighwayDispatchTable;

// P0: Widening Operations
using hwy::MulEvenInt32HighwayDispatchTable;
using hwy::MulEvenUint32HighwayDispatchTable;
using hwy::MulOddInt32HighwayDispatchTable;
using hwy::MulOddUint32HighwayDispatchTable;
using hwy::SumsOf2Int16HighwayDispatchTable;
using hwy::SumsOf2Uint16HighwayDispatchTable;
using hwy::SumsOf4Int8HighwayDispatchTable;
using hwy::SumsOf4Uint8HighwayDispatchTable;

// P0: Additional Comparison Operations
using hwy::IsEitherNaNFloat32HighwayDispatchTable;
using hwy::IsEitherNaNFloat64HighwayDispatchTable;
using hwy::IsNegativeFloat32HighwayDispatchTable;
using hwy::IsNegativeFloat64HighwayDispatchTable;
using hwy::IsNegativeInt32HighwayDispatchTable;

// P1: Extended FMA Operations
using hwy::MulAddSubFloat32HighwayDispatchTable;
using hwy::MulAddSubFloat64HighwayDispatchTable;
using hwy::NegMulSubFloat32HighwayDispatchTable;
using hwy::NegMulSubFloat64HighwayDispatchTable;

// P1: CompressStore Operations
using hwy::CompressStoreFloat32HighwayDispatchTable;
using hwy::CompressStoreFloat64HighwayDispatchTable;
using hwy::CompressStoreInt32HighwayDispatchTable;

// P2: Iota Operations
using hwy::IotaFloat32HighwayDispatchTable;
using hwy::IotaFloat64HighwayDispatchTable;
using hwy::IotaInt32HighwayDispatchTable;
using hwy::IotaInt64HighwayDispatchTable;
using hwy::IotaUint32HighwayDispatchTable;
using hwy::IotaUint64HighwayDispatchTable;

// P2: FirstN Operation
using hwy::FirstNImplHighwayDispatchTable;

// P0: WidenMulAccumulate Operations
using hwy::WidenMulAccumulateInt16HighwayDispatchTable;
using hwy::WidenMulAccumulateUint16HighwayDispatchTable;

// P0: SumsOf8 Operation
using hwy::SumsOf8Uint8HighwayDispatchTable;

// P0: TestBit Operation
using hwy::TestBitInt32HighwayDispatchTable;
using hwy::TestBitInt64HighwayDispatchTable;
using hwy::TestBitUint32HighwayDispatchTable;
using hwy::TestBitUint64HighwayDispatchTable;

// P1: MulSubAdd Operations
using hwy::MulSubAddFloat32HighwayDispatchTable;
using hwy::MulSubAddFloat64HighwayDispatchTable;

// P1: Reverse2, Reverse4, Reverse8 Operations
using hwy::Reverse2Float32HighwayDispatchTable;
using hwy::Reverse2Float64HighwayDispatchTable;
using hwy::Reverse2Int32HighwayDispatchTable;
using hwy::Reverse4Float32HighwayDispatchTable;
using hwy::Reverse4Int32HighwayDispatchTable;
using hwy::Reverse8Uint8HighwayDispatchTable;

// P1: DupEven, DupOdd Operations
using hwy::DupEvenFloat32HighwayDispatchTable;
using hwy::DupEvenInt32HighwayDispatchTable;
using hwy::DupOddFloat32HighwayDispatchTable;
using hwy::DupOddInt32HighwayDispatchTable;

// P1: InterleaveLower, InterleaveUpper Operations
using hwy::InterleaveLowerFloat32HighwayDispatchTable;
using hwy::InterleaveLowerInt32HighwayDispatchTable;
using hwy::InterleaveUpperFloat32HighwayDispatchTable;
using hwy::InterleaveUpperInt32HighwayDispatchTable;

// P1: Mask Logical Operations
using hwy::MaskAndNotUint8HighwayDispatchTable;
using hwy::MaskAndUint8HighwayDispatchTable;
using hwy::MaskNotUint8HighwayDispatchTable;
using hwy::MaskOrUint8HighwayDispatchTable;
using hwy::MaskXorUint8HighwayDispatchTable;

// P2: AddSub Operations
using hwy::AddSubFloat32HighwayDispatchTable;
using hwy::AddSubFloat64HighwayDispatchTable;

// P2: MinMagnitude, MaxMagnitude Operations
using hwy::MaxMagnitudeFloat32HighwayDispatchTable;
using hwy::MaxMagnitudeFloat64HighwayDispatchTable;
using hwy::MinMagnitudeFloat32HighwayDispatchTable;
using hwy::MinMagnitudeFloat64HighwayDispatchTable;

// P2: MaskedLoad, MaskedStore, BlendedStore Operations
using hwy::BlendedStoreFloat32HighwayDispatchTable;
using hwy::BlendedStoreFloat64HighwayDispatchTable;
using hwy::BlendedStoreInt32HighwayDispatchTable;
using hwy::MaskedLoadFloat32HighwayDispatchTable;
using hwy::MaskedLoadFloat64HighwayDispatchTable;
using hwy::MaskedLoadInt32HighwayDispatchTable;
using hwy::MaskedStoreFloat32HighwayDispatchTable;
using hwy::MaskedStoreFloat64HighwayDispatchTable;
using hwy::MaskedStoreInt32HighwayDispatchTable;

// P0: WidenMulPairwiseAdd Operations
using hwy::WidenMulPairwiseAddInt16HighwayDispatchTable;
using hwy::WidenMulPairwiseAddUint16HighwayDispatchTable;

// P1: BroadcastLane Operations
using hwy::BroadcastLaneFloat32HighwayDispatchTable;
using hwy::BroadcastLaneFloat64HighwayDispatchTable;
using hwy::BroadcastLaneInt32HighwayDispatchTable;

// P1: Slide Operations
using hwy::Slide1DownFloat32HighwayDispatchTable;
using hwy::Slide1DownInt32HighwayDispatchTable;
using hwy::Slide1UpFloat32HighwayDispatchTable;
using hwy::Slide1UpInt32HighwayDispatchTable;

// P1: Concat Operations
using hwy::ConcatEvenFloat32HighwayDispatchTable;
using hwy::ConcatEvenInt32HighwayDispatchTable;
using hwy::ConcatLowerUpperFloat32HighwayDispatchTable;
using hwy::ConcatLowerUpperInt32HighwayDispatchTable;
using hwy::ConcatOddFloat32HighwayDispatchTable;
using hwy::ConcatOddInt32HighwayDispatchTable;
using hwy::ConcatUpperLowerFloat32HighwayDispatchTable;
using hwy::ConcatUpperLowerInt32HighwayDispatchTable;

// P1: Mask Utility Operations
using hwy::FindKnownFirstTrueImplHighwayDispatchTable;
using hwy::FindKnownLastTrueImplHighwayDispatchTable;
using hwy::LoadMaskBitsImplHighwayDispatchTable;
using hwy::StoreMaskBitsImplHighwayDispatchTable;

// P1: CompressBlendedStore and CompressNot Operations
using hwy::CompressBlendedStoreFloat32HighwayDispatchTable;
using hwy::CompressBlendedStoreInt32HighwayDispatchTable;
using hwy::CompressNotFloat32HighwayDispatchTable;
using hwy::CompressNotInt32HighwayDispatchTable;

// P1: InterleaveEven and InterleaveOdd Operations
using hwy::InterleaveEvenFloat32HighwayDispatchTable;
using hwy::InterleaveEvenInt32HighwayDispatchTable;
using hwy::InterleaveOddFloat32HighwayDispatchTable;
using hwy::InterleaveOddInt32HighwayDispatchTable;

// P1: Shuffle Operations
using hwy::Shuffle0123Float32HighwayDispatchTable;
using hwy::Shuffle0123Int32HighwayDispatchTable;
using hwy::Shuffle01Float64HighwayDispatchTable;
using hwy::Shuffle01Int64HighwayDispatchTable;
using hwy::Shuffle1032Float32HighwayDispatchTable;
using hwy::Shuffle1032Int32HighwayDispatchTable;
using hwy::Shuffle10Float64HighwayDispatchTable;
using hwy::Shuffle10Int64HighwayDispatchTable;
using hwy::Shuffle2301Float32HighwayDispatchTable;
using hwy::Shuffle2301Int32HighwayDispatchTable;

// P1: TableLookup Operations
using hwy::TableLookupBytesUint8HighwayDispatchTable;
using hwy::TableLookupLanesFloat32HighwayDispatchTable;
using hwy::TableLookupLanesInt32HighwayDispatchTable;

// P1: Mask Set Operations
using hwy::SetAtOrAfterFirstImplHighwayDispatchTable;
using hwy::SetAtOrBeforeFirstImplHighwayDispatchTable;
using hwy::SetBeforeFirstImplHighwayDispatchTable;
using hwy::SetOnlyFirstImplHighwayDispatchTable;

// P1: Masked Reduction Operations
using hwy::MaskedReduceMaxFloat32HighwayDispatchTable;
using hwy::MaskedReduceMaxFloat64HighwayDispatchTable;
using hwy::MaskedReduceMaxInt32HighwayDispatchTable;
using hwy::MaskedReduceMinFloat32HighwayDispatchTable;
using hwy::MaskedReduceMinFloat64HighwayDispatchTable;
using hwy::MaskedReduceMinInt32HighwayDispatchTable;
using hwy::MaskedReduceSumFloat32HighwayDispatchTable;
using hwy::MaskedReduceSumFloat64HighwayDispatchTable;
using hwy::MaskedReduceSumInt32HighwayDispatchTable;

// Remaining P1 Operations
using hwy::CompressBitsFloat32HighwayDispatchTable;
using hwy::CompressBitsInt32HighwayDispatchTable;
using hwy::CompressBitsStoreFloat32HighwayDispatchTable;
using hwy::CompressBitsStoreInt32HighwayDispatchTable;
using hwy::LoadExpandFloat32HighwayDispatchTable;
using hwy::LoadExpandInt32HighwayDispatchTable;
using hwy::PairwiseSubFloat32HighwayDispatchTable;
using hwy::PairwiseSubInt32HighwayDispatchTable;
using hwy::SumsOfAdjQuadAbsDiffUint8HighwayDispatchTable;
using hwy::TwoTablesLookupLanesFloat32HighwayDispatchTable;
using hwy::TwoTablesLookupLanesInt32HighwayDispatchTable;

// P2: Math Functions
using hwy::CbrtFloat32HighwayDispatchTable;
using hwy::CbrtFloat64HighwayDispatchTable;
using hwy::ErfcFloat32HighwayDispatchTable;
using hwy::ErfcFloat64HighwayDispatchTable;
using hwy::ErfFloat32HighwayDispatchTable;
using hwy::ErfFloat64HighwayDispatchTable;

// P2: Generation Operations
using hwy::IndicesFromNotVecImplHighwayDispatchTable;
using hwy::IndicesFromVecImplHighwayDispatchTable;

// P2: Type Conversions
using hwy::PromoteEvenToFloat64HighwayDispatchTable;
using hwy::PromoteEvenToInt32HighwayDispatchTable;
using hwy::PromoteLowerToFloat64HighwayDispatchTable;
using hwy::PromoteLowerToInt32HighwayDispatchTable;
using hwy::PromoteLowerToInt64HighwayDispatchTable;
using hwy::PromoteOddToFloat64HighwayDispatchTable;
using hwy::PromoteOddToInt32HighwayDispatchTable;
using hwy::PromoteUpperToFloat64HighwayDispatchTable;
using hwy::PromoteUpperToInt32HighwayDispatchTable;

// P2: Additional Arithmetic
using hwy::ModInt32HighwayDispatchTable;
using hwy::ModInt64HighwayDispatchTable;
using hwy::SaturatedAbsInt16HighwayDispatchTable;
using hwy::SaturatedAbsInt8HighwayDispatchTable;
using hwy::SaturatedNegInt16HighwayDispatchTable;
using hwy::SaturatedNegInt8HighwayDispatchTable;

// P2: Bitwise Operations
using hwy::HighestSetBitIndexUint32HighwayDispatchTable;
using hwy::HighestSetBitIndexUint64HighwayDispatchTable;
using hwy::Or3Int32HighwayDispatchTable;
using hwy::Or3Int64HighwayDispatchTable;
using hwy::OrAndInt32HighwayDispatchTable;
using hwy::OrAndInt64HighwayDispatchTable;
using hwy::ReverseBitsUint32HighwayDispatchTable;
using hwy::ReverseBitsUint64HighwayDispatchTable;
using hwy::Xor3Int32HighwayDispatchTable;
using hwy::Xor3Int64HighwayDispatchTable;

// P2: Memory Operations
using hwy::GatherOffsetFloat32HighwayDispatchTable;
using hwy::GatherOffsetInt32HighwayDispatchTable;
using hwy::LoadDup128Float32HighwayDispatchTable;
using hwy::LoadDup128Float64HighwayDispatchTable;
using hwy::MaskedGatherIndexFloat32HighwayDispatchTable;
using hwy::MaskedGatherIndexInt32HighwayDispatchTable;
using hwy::MaskedScatterIndexFloat32HighwayDispatchTable;
using hwy::MaskedScatterIndexInt32HighwayDispatchTable;
using hwy::SafeCopyNFloat32HighwayDispatchTable;
using hwy::SafeCopyNInt32HighwayDispatchTable;
using hwy::SafeFillNFloat32HighwayDispatchTable;
using hwy::SafeFillNInt32HighwayDispatchTable;
using hwy::ScatterOffsetFloat32HighwayDispatchTable;
using hwy::ScatterOffsetInt32HighwayDispatchTable;

// P2: Special Operations
using hwy::GetExponentFloat32HighwayDispatchTable;
using hwy::GetExponentFloat64HighwayDispatchTable;
using hwy::InfFloat32HighwayDispatchTable;
using hwy::InfFloat64HighwayDispatchTable;
using hwy::MulByPow2Float32HighwayDispatchTable;
using hwy::MulByPow2Float64HighwayDispatchTable;
using hwy::NaNFloat32HighwayDispatchTable;
using hwy::NaNFloat64HighwayDispatchTable;
using hwy::SignBitFloat32HighwayDispatchTable;
using hwy::SignBitFloat64HighwayDispatchTable;

// P3: Complex Number Operations
using hwy::ComplexConjFloat32HighwayDispatchTable;
using hwy::ComplexConjFloat64HighwayDispatchTable;
using hwy::MulComplexAddFloat32HighwayDispatchTable;
using hwy::MulComplexFloat32HighwayDispatchTable;
using hwy::MulComplexFloat64HighwayDispatchTable;

// P3: Saturation Operations
using hwy::SaturatedAddInt8HighwayDispatchTable;
using hwy::SaturatedAddUint16HighwayDispatchTable;
using hwy::SaturatedSubInt8HighwayDispatchTable;
using hwy::SaturatedSubUint16HighwayDispatchTable;

// P3: Block Operations
using hwy::CombineShiftRightLanesFloat32HighwayDispatchTable;
using hwy::SlideDownBlocksFloat32HighwayDispatchTable;
using hwy::SlideUpBlocksFloat32HighwayDispatchTable;

// P3: Additional Masked Operations
using hwy::MaskedSqrtFloat32HighwayDispatchTable;
using hwy::MaskedSqrtFloat64HighwayDispatchTable;
using hwy::ZeroIfNegativeFloat32HighwayDispatchTable;
using hwy::ZeroIfNegativeFloat64HighwayDispatchTable;

// P2: Additional Type Conversion Operations
using hwy::ConvertInRangeToFloat32Int32HighwayDispatchTable;
using hwy::ConvertInRangeToFloat64Int64HighwayDispatchTable;
using hwy::OrderedTruncate2ToInt32Int16HighwayDispatchTable;
using hwy::OrderedTruncate2ToUint32Uint16HighwayDispatchTable;
using hwy::ReorderDemote2ToInt32Int16HighwayDispatchTable;
using hwy::ReorderDemote2ToUint32Uint16HighwayDispatchTable;
using hwy::ResizeBitCastFloat32Uint32HighwayDispatchTable;
using hwy::ResizeBitCastFloat64Uint64HighwayDispatchTable;
using hwy::ResizeBitCastUint32Float32HighwayDispatchTable;
using hwy::ResizeBitCastUint64Float64HighwayDispatchTable;

// P2: Additional Special Operations
using hwy::GetBiasedExponentFloat32HighwayDispatchTable;
using hwy::GetBiasedExponentFloat64HighwayDispatchTable;
using hwy::MulByFloorPow2Float32HighwayDispatchTable;
using hwy::MulByFloorPow2Float64HighwayDispatchTable;
using hwy::MulFixedPoint15Int16HighwayDispatchTable;
using hwy::MulRoundInt16HighwayDispatchTable;
using hwy::MulRoundInt32HighwayDispatchTable;
using hwy::RoundingShiftRightInt16HighwayDispatchTable;
using hwy::RoundingShiftRightInt32HighwayDispatchTable;

// P3: Additional Complex Number Operations
using hwy::MulComplexConjAddFloat32HighwayDispatchTable;
using hwy::MulComplexConjAddFloat64HighwayDispatchTable;
using hwy::MulComplexConjFloat32HighwayDispatchTable;
using hwy::MulComplexConjFloat64HighwayDispatchTable;

// P3: Per-Lane Block Shuffle
using hwy::Per4LaneBlockShuffleFloat32HighwayDispatchTable;
using hwy::Per4LaneBlockShuffleInt32HighwayDispatchTable;

// P3: Additional Masked Operations
using hwy::MaskedReciprocalFloat32HighwayDispatchTable;
using hwy::MaskedReciprocalFloat64HighwayDispatchTable;
using hwy::MaskedSatAddInt16HighwayDispatchTable;
using hwy::MaskedSatAddInt8HighwayDispatchTable;
using hwy::MaskedSatSubInt16HighwayDispatchTable;
using hwy::MaskedSatSubInt8HighwayDispatchTable;
using hwy::MaskedShiftLeftInt32HighwayDispatchTable;
using hwy::MaskedShiftLeftInt64HighwayDispatchTable;
using hwy::MaskedShiftRightInt32HighwayDispatchTable;
using hwy::MaskedShiftRightInt64HighwayDispatchTable;

// P3: Masked Comparison Operations
using hwy::MaskedEqFloat32HighwayDispatchTable;
using hwy::MaskedEqInt32HighwayDispatchTable;
using hwy::MaskedGeFloat32HighwayDispatchTable;
using hwy::MaskedGeInt32HighwayDispatchTable;
using hwy::MaskedGtFloat32HighwayDispatchTable;
using hwy::MaskedGtInt32HighwayDispatchTable;
using hwy::MaskedLeFloat32HighwayDispatchTable;
using hwy::MaskedLeInt32HighwayDispatchTable;
using hwy::MaskedLtFloat32HighwayDispatchTable;
using hwy::MaskedLtInt32HighwayDispatchTable;
using hwy::MaskedNeFloat32HighwayDispatchTable;
using hwy::MaskedNeInt32HighwayDispatchTable;

// P3.2: Cryptographic Operations
using hwy::AESInvMixColumnsImplHighwayDispatchTable;
using hwy::AESKeyGenAssistImplHighwayDispatchTable;
using hwy::AESLastRoundImplHighwayDispatchTable;
using hwy::AESLastRoundInvImplHighwayDispatchTable;
using hwy::AESRoundImplHighwayDispatchTable;
using hwy::AESRoundInvImplHighwayDispatchTable;
using hwy::CLMulLowerImplHighwayDispatchTable;
using hwy::CLMulUpperImplHighwayDispatchTable;

// P3.3: Random Number Generation
using hwy::Random32ImplHighwayDispatchTable;
using hwy::Random64ImplHighwayDispatchTable;
using hwy::RandomFloatImplHighwayDispatchTable;
using hwy::RandomStateInitImplHighwayDispatchTable;

// P3.4: Bit Packing
using hwy::PackBitsImplHighwayDispatchTable;
using hwy::UnpackBitsImplHighwayDispatchTable;

// P3.6: Algorithm Operations
using hwy::FindIfGreaterThanFloat32HighwayDispatchTable;
using hwy::FindIfGreaterThanInt32HighwayDispatchTable;
using hwy::GenerateFloat32HighwayDispatchTable;
using hwy::GenerateInt32HighwayDispatchTable;
using hwy::ReplaceFloat32HighwayDispatchTable;
using hwy::ReplaceIfGreaterThanFloat32HighwayDispatchTable;
using hwy::ReplaceIfGreaterThanInt32HighwayDispatchTable;
using hwy::ReplaceInt32HighwayDispatchTable;

// P3.5: Image Processing Operations
using hwy::BoxBlur3x3ImplHighwayDispatchTable;
using hwy::Convolve3x3ImplHighwayDispatchTable;
using hwy::Downsample2xImplHighwayDispatchTable;
using hwy::GaussianBlur3x3ImplHighwayDispatchTable;
using hwy::GrayscaleImplHighwayDispatchTable;
using hwy::ImageAddImplHighwayDispatchTable;
using hwy::ImageClampImplHighwayDispatchTable;
using hwy::ImageCopyImplHighwayDispatchTable;
using hwy::ImageFillImplHighwayDispatchTable;
using hwy::ImageMulImplHighwayDispatchTable;
using hwy::ImageScaleImplHighwayDispatchTable;
using hwy::ImageSubImplHighwayDispatchTable;
using hwy::SharpenImplHighwayDispatchTable;
using hwy::SobelEdgeImplHighwayDispatchTable;
using hwy::ThresholdImplHighwayDispatchTable;
using hwy::Upsample2xImplHighwayDispatchTable;

// Gap Operations: Additional SIMD Operations
using hwy::BitwiseIfThenElseInt32HighwayDispatchTable;
using hwy::CeilIntFloat32HighwayDispatchTable;
using hwy::FloorIntFloat32HighwayDispatchTable;
using hwy::IfNegativeThenElseZeroFloat32HighwayDispatchTable;
using hwy::IfNegativeThenZeroElseFloat32HighwayDispatchTable;
using hwy::InterleaveWholeLowerFloat32HighwayDispatchTable;
using hwy::InterleaveWholeUpperFloat32HighwayDispatchTable;
using hwy::MaskedModOrInt32HighwayDispatchTable;
using hwy::MaskedMulAddFloat32HighwayDispatchTable;
using hwy::MaskedNegMulAddFloat32HighwayDispatchTable;
using hwy::MaskFalseImplHighwayDispatchTable;
using hwy::OddEvenFloat32HighwayDispatchTable;
using hwy::OddEvenInt32HighwayDispatchTable;
using hwy::SetMaskImplHighwayDispatchTable;
using hwy::TruncateStoreInt32ToInt16HighwayDispatchTable;

// Final gap operations
using hwy::CombineMasksImplHighwayDispatchTable;
using hwy::DemoteMaskToUint16ToUint8HighwayDispatchTable;
using hwy::F64ToBF16ImplHighwayDispatchTable;
using hwy::F64ToF16ImplHighwayDispatchTable;
using hwy::IfNegativeThenNegOrUndefIfZeroFloat32HighwayDispatchTable;
using hwy::InsertIntoUpperFloat32HighwayDispatchTable;
using hwy::LowerHalfOfMaskImplHighwayDispatchTable;
using hwy::MaskedAbsOrFloat32HighwayDispatchTable;
using hwy::MaskedGatherIndexOrFloat32HighwayDispatchTable;
using hwy::MaskedIsNaNFloat32HighwayDispatchTable;
using hwy::MaskedMulFixedPoint15Int16HighwayDispatchTable;
using hwy::MaskedSetOrFloat32HighwayDispatchTable;
using hwy::MaskedWidenMulPairwiseAddInt16ToInt32HighwayDispatchTable;
using hwy::MatMulFloat32HighwayDispatchTable;
using hwy::MatVecMulFloat32HighwayDispatchTable;
using hwy::Per2LaneBlockShuffleFloat32HighwayDispatchTable;
using hwy::PromoteMaskToUint8ToUint16HighwayDispatchTable;
using hwy::SumsOf8AbsDiffUint8HighwayDispatchTable;
using hwy::UpperHalfOfMaskImplHighwayDispatchTable;
using hwy::ZeroExtendResizeBitCastUint8ToUint32HighwayDispatchTable;

// Priority 1: Arithmetic Operations

void Add(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddFloat32)(out, a, b, count);
}

void Add(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddFloat64)(out, a, b, count);
}

void Add(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddInt32)(out, a, b, count);
}

void Add(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddInt64)(out, a, b, count);
}

void Sub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubFloat32)(out, a, b, count);
}

void Sub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubFloat64)(out, a, b, count);
}

void Sub(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubInt32)(out, a, b, count);
}

void Sub(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubInt64)(out, a, b, count);
}

void Mul(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MulFloat32)(out, a, b, count);
}

void Mul(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MulFloat64)(out, a, b, count);
}

void Mul(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MulInt32)(out, a, b, count);
}

void Mul(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MulInt64)(out, a, b, count);
}

void Div(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(DivFloat32)(out, a, b, count);
}

void Div(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(DivFloat64)(out, a, b, count);
}

void Neg(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegFloat32)(out, a, count);
}

void Neg(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegFloat64)(out, a, count);
}

void Neg(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegInt32)(out, a, count);
}

void Neg(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegInt64)(out, a, count);
}

void Abs(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AbsFloat32)(out, a, count);
}

void Abs(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AbsFloat64)(out, a, count);
}

void Abs(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AbsInt32)(out, a, count);
}

void Abs(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AbsInt64)(out, a, count);
}

// Priority 1: FMA Operations

void MulAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulAddFloat32)(out, a, b, c, count);
}

void MulAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulAddFloat64)(out, a, b, c, count);
}

void MulSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulSubFloat32)(out, a, b, c, count);
}

void MulSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulSubFloat64)(out, a, b, c, count);
}

void NegMulAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegMulAddFloat32)(out, a, b, c, count);
}

void NegMulAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegMulAddFloat64)(out, a, b, c, count);
}

// Priority 1: MinMax Operations

void Min(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinFloat32)(out, a, b, count);
}

void Min(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinFloat64)(out, a, b, count);
}

void Min(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinInt32)(out, a, b, count);
}

void Min(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinInt64)(out, a, b, count);
}

void Max(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxFloat32)(out, a, b, count);
}

void Max(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxFloat64)(out, a, b, count);
}

void Max(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxInt32)(out, a, b, count);
}

void Max(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxInt64)(out, a, b, count);
}

void Clamp(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, float lo, float hi, size_t count) {
    HWY_DYNAMIC_DISPATCH(ClampFloat32)(out, a, lo, hi, count);
}

void Clamp(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, double lo, double hi,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(ClampFloat64)(out, a, lo, hi, count);
}

void Clamp(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, int32_t lo, int32_t hi,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(ClampInt32)(out, a, lo, hi, count);
}

void Clamp(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, int64_t lo, int64_t hi,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(ClampInt64)(out, a, lo, hi, count);
}

// Priority 1: Math Operations

void Sqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SqrtFloat32)(out, a, count);
}

void Sqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SqrtFloat64)(out, a, count);
}

void Rsqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(RsqrtFloat32)(out, a, count);
}

void Rsqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(RsqrtFloat64)(out, a, count);
}

void Exp(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ExpFloat32)(out, a, count);
}

void Exp(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ExpFloat64)(out, a, count);
}

void Log(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(LogFloat32)(out, a, count);
}

void Log(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(LogFloat64)(out, a, count);
}

void Sin(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SinFloat32)(out, a, count);
}

void Sin(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SinFloat64)(out, a, count);
}

void Cos(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CosFloat32)(out, a, count);
}

void Cos(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CosFloat64)(out, a, count);
}

void Tanh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TanhFloat32)(out, a, count);
}

void Tanh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TanhFloat64)(out, a, count);
}

// Priority 1: Comparison Operations

void Eq(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(EqFloat32)(out, a, b, count);
}

void Eq(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(EqFloat64)(out, a, b, count);
}

void Eq(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(EqInt32)(out, a, b, count);
}

void Ne(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(NeFloat32)(out, a, b, count);
}

void Ne(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(NeFloat64)(out, a, b, count);
}

void Ne(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(NeInt32)(out, a, b, count);
}

void Lt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(LtFloat32)(out, a, b, count);
}

void Lt(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(LtFloat64)(out, a, b, count);
}

void Lt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(LtInt32)(out, a, b, count);
}

void Le(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(LeFloat32)(out, a, b, count);
}

void Le(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(LeFloat64)(out, a, b, count);
}

void Le(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(LeInt32)(out, a, b, count);
}

void Gt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(GtFloat32)(out, a, b, count);
}

void Gt(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(GtFloat64)(out, a, b, count);
}

void Gt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(GtInt32)(out, a, b, count);
}

void Ge(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(GeFloat32)(out, a, b, count);
}

void Ge(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(GeFloat64)(out, a, b, count);
}

void Ge(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
        size_t count) {
    HWY_DYNAMIC_DISPATCH(GeInt32)(out, a, b, count);
}

// Priority 1: Reduction Operations

float ReduceSum(const float* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceSumFloat32)(a, count);
}

double ReduceSum(const double* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceSumFloat64)(a, count);
}

int32_t ReduceSum(const int32_t* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceSumInt32)(a, count);
}

int64_t ReduceSum(const int64_t* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceSumInt64)(a, count);
}

float ReduceMin(const float* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMinFloat32)(a, count);
}

double ReduceMin(const double* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMinFloat64)(a, count);
}

int32_t ReduceMin(const int32_t* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMinInt32)(a, count);
}

int64_t ReduceMin(const int64_t* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMinInt64)(a, count);
}

float ReduceMax(const float* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMaxFloat32)(a, count);
}

double ReduceMax(const double* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMaxFloat64)(a, count);
}

int32_t ReduceMax(const int32_t* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMaxInt32)(a, count);
}

int64_t ReduceMax(const int64_t* HWY_RESTRICT a, size_t count) {
    return HWY_DYNAMIC_DISPATCH(ReduceMaxInt64)(a, count);
}

float Dot(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t count) {
    return HWY_DYNAMIC_DISPATCH(DotFloat32)(a, b, count);
}

double Dot(const double* HWY_RESTRICT a, const double* HWY_RESTRICT b, size_t count) {
    return HWY_DYNAMIC_DISPATCH(DotFloat64)(a, b, count);
}

// Priority 1: Select Operations

void Select(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
            const float* HWY_RESTRICT yes, const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(SelectFloat32)(out, mask, yes, no, count);
}

void Select(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
            const double* HWY_RESTRICT yes, const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(SelectFloat64)(out, mask, yes, no, count);
}

void Select(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
            const int32_t* HWY_RESTRICT yes, const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(SelectInt32)(out, mask, yes, no, count);
}

// Priority 2: Extended Math Operations

void Exp2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Exp2Float32)(out, a, count);
}

void Exp2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Exp2Float64)(out, a, count);
}

void Log2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Log2Float32)(out, a, count);
}

void Log2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Log2Float64)(out, a, count);
}

void Log10(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Log10Float32)(out, a, count);
}

void Log10(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Log10Float64)(out, a, count);
}

void Sinh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SinhFloat32)(out, a, count);
}

void Sinh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SinhFloat64)(out, a, count);
}

void Cosh(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CoshFloat32)(out, a, count);
}

void Cosh(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CoshFloat64)(out, a, count);
}

// Priority 2: Inverse Trigonometric Operations

void Asin(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AsinFloat32)(out, a, count);
}

void Asin(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AsinFloat64)(out, a, count);
}

void Acos(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AcosFloat32)(out, a, count);
}

void Acos(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AcosFloat64)(out, a, count);
}

void Atan(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AtanFloat32)(out, a, count);
}

void Atan(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(AtanFloat64)(out, a, count);
}

void Atan2(float* HWY_RESTRICT out, const float* HWY_RESTRICT y, const float* HWY_RESTRICT x,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(Atan2Float32)(out, y, x, count);
}

void Atan2(double* HWY_RESTRICT out, const double* HWY_RESTRICT y, const double* HWY_RESTRICT x,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(Atan2Float64)(out, y, x, count);
}

// Priority 2: Rounding Operations

void Round(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(RoundFloat32)(out, a, count);
}

void Round(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(RoundFloat64)(out, a, count);
}

void Floor(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(FloorFloat32)(out, a, count);
}

void Floor(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(FloorFloat64)(out, a, count);
}

void Ceil(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CeilFloat32)(out, a, count);
}

void Ceil(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CeilFloat64)(out, a, count);
}

void Trunc(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TruncFloat32)(out, a, count);
}

void Trunc(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TruncFloat64)(out, a, count);
}

// Priority 2: Bitwise Operations

void BitwiseAnd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseAndInt32)(out, a, b, count);
}

void BitwiseAnd(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                const int64_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseAndInt64)(out, a, b, count);
}

void BitwiseOr(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseOrInt32)(out, a, b, count);
}

void BitwiseOr(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
               const int64_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseOrInt64)(out, a, b, count);
}

void BitwiseXor(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseXorInt32)(out, a, b, count);
}

void BitwiseXor(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                const int64_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseXorInt64)(out, a, b, count);
}

void BitwiseNot(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseNotInt32)(out, a, count);
}

void BitwiseNot(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseNotInt64)(out, a, count);
}

// =============================================================================
// Type-Generic Dispatch Functions
// =============================================================================

Result<void> DispatchAdd(void* out, const void* a, const void* b, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Add(static_cast<float*>(out), static_cast<const float*>(a), static_cast<const float*>(b),
            count);
        return {};
    case ScalarType::kFloat64:
        Add(static_cast<double*>(out), static_cast<const double*>(a), static_cast<const double*>(b),
            count);
        return {};
    case ScalarType::kInt32:
        Add(static_cast<int32_t*>(out), static_cast<const int32_t*>(a),
            static_cast<const int32_t*>(b), count);
        return {};
    case ScalarType::kInt64:
        Add(static_cast<int64_t*>(out), static_cast<const int64_t*>(a),
            static_cast<const int64_t*>(b), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchAdd: unsupported type");
    }
}

Result<void> DispatchSub(void* out, const void* a, const void* b, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Sub(static_cast<float*>(out), static_cast<const float*>(a), static_cast<const float*>(b),
            count);
        return {};
    case ScalarType::kFloat64:
        Sub(static_cast<double*>(out), static_cast<const double*>(a), static_cast<const double*>(b),
            count);
        return {};
    case ScalarType::kInt32:
        Sub(static_cast<int32_t*>(out), static_cast<const int32_t*>(a),
            static_cast<const int32_t*>(b), count);
        return {};
    case ScalarType::kInt64:
        Sub(static_cast<int64_t*>(out), static_cast<const int64_t*>(a),
            static_cast<const int64_t*>(b), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchSub: unsupported type");
    }
}

Result<void> DispatchMul(void* out, const void* a, const void* b, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Mul(static_cast<float*>(out), static_cast<const float*>(a), static_cast<const float*>(b),
            count);
        return {};
    case ScalarType::kFloat64:
        Mul(static_cast<double*>(out), static_cast<const double*>(a), static_cast<const double*>(b),
            count);
        return {};
    case ScalarType::kInt32:
        Mul(static_cast<int32_t*>(out), static_cast<const int32_t*>(a),
            static_cast<const int32_t*>(b), count);
        return {};
    case ScalarType::kInt64:
        Mul(static_cast<int64_t*>(out), static_cast<const int64_t*>(a),
            static_cast<const int64_t*>(b), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchMul: unsupported type");
    }
}

Result<void> DispatchDiv(void* out, const void* a, const void* b, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Div(static_cast<float*>(out), static_cast<const float*>(a), static_cast<const float*>(b),
            count);
        return {};
    case ScalarType::kFloat64:
        Div(static_cast<double*>(out), static_cast<const double*>(a), static_cast<const double*>(b),
            count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType,
                     "DispatchDiv: unsupported type (integers not supported)");
    }
}

Result<void> DispatchNeg(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Neg(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Neg(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    case ScalarType::kInt32:
        Neg(static_cast<int32_t*>(out), static_cast<const int32_t*>(a), count);
        return {};
    case ScalarType::kInt64:
        Neg(static_cast<int64_t*>(out), static_cast<const int64_t*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchNeg: unsupported type");
    }
}

Result<void> DispatchAbs(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Abs(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Abs(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    case ScalarType::kInt32:
        Abs(static_cast<int32_t*>(out), static_cast<const int32_t*>(a), count);
        return {};
    case ScalarType::kInt64:
        Abs(static_cast<int64_t*>(out), static_cast<const int64_t*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchAbs: unsupported type");
    }
}

Result<void> DispatchSqrt(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Sqrt(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Sqrt(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchSqrt: unsupported type (floats only)");
    }
}

Result<void> DispatchExp(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Exp(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Exp(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchExp: unsupported type (floats only)");
    }
}

Result<void> DispatchLog(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Log(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Log(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchLog: unsupported type (floats only)");
    }
}

Result<void> DispatchSin(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Sin(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Sin(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchSin: unsupported type (floats only)");
    }
}

Result<void> DispatchCos(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Cos(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Cos(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchCos: unsupported type (floats only)");
    }
}

Result<void> DispatchTanh(void* out, const void* a, size_t count, ScalarType dtype) {
    switch (dtype) {
    case ScalarType::kFloat32:
        Tanh(static_cast<float*>(out), static_cast<const float*>(a), count);
        return {};
    case ScalarType::kFloat64:
        Tanh(static_cast<double*>(out), static_cast<const double*>(a), count);
        return {};
    default:
        return Error(ErrorCode::kUnsupportedType, "DispatchTanh: unsupported type (floats only)");
    }
}

// =============================================================================
// Priority 3: Special Value Checks
// =============================================================================

void IsNaN(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsNaNFloat32)(out, a, count);
}

void IsNaN(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsNaNFloat64)(out, a, count);
}

void IsInf(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsInfFloat32)(out, a, count);
}

void IsInf(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsInfFloat64)(out, a, count);
}

void IsFinite(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsFiniteFloat32)(out, a, count);
}

void IsFinite(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsFiniteFloat64)(out, a, count);
}

// =============================================================================
// Priority 3: Variable Shift Operations
// =============================================================================

void ShiftLeftVar(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                  const int32_t* HWY_RESTRICT shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(ShiftLeftVarInt32)(out, a, shift, count);
}

void ShiftLeftVar(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                  const int64_t* HWY_RESTRICT shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(ShiftLeftVarInt64)(out, a, shift, count);
}

void ShiftLeftVar(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                  const uint32_t* HWY_RESTRICT shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(ShiftLeftVarUint32)(out, a, shift, count);
}

void ShiftRightVar(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                   const int32_t* HWY_RESTRICT shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(ShiftRightVarInt32)(out, a, shift, count);
}

void ShiftRightVar(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a,
                   const int64_t* HWY_RESTRICT shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(ShiftRightVarInt64)(out, a, shift, count);
}

void ShiftRightVar(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
                   const uint32_t* HWY_RESTRICT shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(ShiftRightVarUint32)(out, a, shift, count);
}

// =============================================================================
// Priority 3: Type Conversions
// =============================================================================

void PromoteTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteInt16ToInt32)(out, a, count);
}

void PromoteTo(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteUint8ToInt32)(out, a, count);
}

void PromoteTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteFloat32ToFloat64)(out, a, count);
}

void DemoteTo(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(DemoteInt32ToInt16)(out, a, count);
}

void DemoteTo(float* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(DemoteFloat64ToFloat32)(out, a, count);
}

void ConvertTo(float* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConvertInt32ToFloat32)(out, a, count);
}

void ConvertTo(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConvertFloat32ToInt32)(out, a, count);
}

// =============================================================================
// Priority 3: Gather/Scatter
// =============================================================================

void Gather(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
            const int32_t* HWY_RESTRICT indices, size_t count) {
    HWY_DYNAMIC_DISPATCH(GatherFloat32)(out, base, indices, count);
}

void Gather(double* HWY_RESTRICT out, const double* HWY_RESTRICT base,
            const int64_t* HWY_RESTRICT indices, size_t count) {
    HWY_DYNAMIC_DISPATCH(GatherFloat64)(out, base, indices, count);
}

void Gather(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
            const int32_t* HWY_RESTRICT indices, size_t count) {
    HWY_DYNAMIC_DISPATCH(GatherInt32)(out, base, indices, count);
}

void Scatter(const float* HWY_RESTRICT values, float* HWY_RESTRICT base,
             const int32_t* HWY_RESTRICT indices, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScatterFloat32)(values, base, indices, count);
}

void Scatter(const double* HWY_RESTRICT values, double* HWY_RESTRICT base,
             const int64_t* HWY_RESTRICT indices, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScatterFloat64)(values, base, indices, count);
}

void Scatter(const int32_t* HWY_RESTRICT values, int32_t* HWY_RESTRICT base,
             const int32_t* HWY_RESTRICT indices, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScatterInt32)(values, base, indices, count);
}

// =============================================================================
// Priority 3: Horizontal Reductions (scalar fallback for now)
// =============================================================================

void SumOfLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    float sum = 0;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i];
    }
    out[0] = sum;
}

void SumOfLanes(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    double sum = 0;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i];
    }
    out[0] = sum;
}

void SumOfLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    int32_t sum = 0;
    for (size_t i = 0; i < count; ++i) {
        sum += a[i];
    }
    out[0] = sum;
}

void PairwiseAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PairwiseAddFloat32)(out, a, count);
}

void PairwiseAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PairwiseAddFloat64)(out, a, count);
}

void PairwiseAdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PairwiseAddInt32)(out, a, count);
}

// =============================================================================
// Priority 3: Saturation Operations
// =============================================================================

void SaturatedAdd(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedAddInt16)(out, a, b, count);
}

void SaturatedAdd(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                  const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedAddUint8)(out, a, b, count);
}

void SaturatedSub(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedSubInt16)(out, a, b, count);
}

void SaturatedSub(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                  const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedSubUint8)(out, a, b, count);
}

// =============================================================================
// Priority 3: Broadcast
// =============================================================================

void Broadcast(float* HWY_RESTRICT out, float value, size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastFloat32)(out, value, count);
}

void Broadcast(double* HWY_RESTRICT out, double value, size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastFloat64)(out, value, count);
}

void Broadcast(int32_t* HWY_RESTRICT out, int32_t value, size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastInt32)(out, value, count);
}

void Broadcast(int64_t* HWY_RESTRICT out, int64_t value, size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastInt64)(out, value, count);
}

// =============================================================================
// Priority 3: Compress/Expand
// =============================================================================

size_t Compress(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressFloat32)(out, a, mask, count);
}

size_t Compress(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressFloat64)(out, a, mask, count);
}

size_t Compress(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressInt32)(out, a, mask, count);
}

void Expand(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT mask,
            size_t count) {
    // Scalar fallback: read from 'a' into positions where mask is true
    size_t src_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = a[src_idx++];
        }
    }
}

void Expand(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
            const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t src_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = a[src_idx++];
        }
    }
}

void Expand(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
            const uint8_t* HWY_RESTRICT mask, size_t count) {
    size_t src_idx = 0;
    for (size_t i = 0; i < count; ++i) {
        if (mask[i]) {
            out[i] = a[src_idx++];
        }
    }
}

// =============================================================================
// Priority 3: Interleave/Deinterleave
// =============================================================================

void Interleave(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveFloat32)(out, a, b, count);
}

void Interleave(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                const double* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveFloat64)(out, a, b, count);
}

void Deinterleave(float* HWY_RESTRICT out_a, float* HWY_RESTRICT out_b,
                  const float* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(DeinterleaveFloat32)(out_a, out_b, interleaved, count);
}

void Deinterleave(double* HWY_RESTRICT out_a, double* HWY_RESTRICT out_b,
                  const double* HWY_RESTRICT interleaved, size_t count) {
    // Scalar fallback for double
    for (size_t i = 0; i < count; ++i) {
        out_a[i] = interleaved[2 * i];
        out_b[i] = interleaved[2 * i + 1];
    }
}

// =============================================================================
// Priority 3: 8/16-bit Integer Operations
// =============================================================================

void Add(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddInt8)(out, a, b, count);
}

void Sub(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubInt8)(out, a, b, count);
}

void Mul(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count) {
    // Scalar fallback - no 8-bit SIMD mul in Highway
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<int8_t>(a[i] * b[i]);
    }
}

void Min(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinInt8)(out, a, b, count);
}

void Max(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, const int8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxInt8)(out, a, b, count);
}

void Add(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddInt16)(out, a, b, count);
}

void Sub(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubInt16)(out, a, b, count);
}

void Mul(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MulInt16)(out, a, b, count);
}

void Min(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinInt16)(out, a, b, count);
}

void Max(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxInt16)(out, a, b, count);
}

void Add(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddUint8)(out, a, b, count);
}

void Sub(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubUint8)(out, a, b, count);
}

void Mul(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count) {
    // Scalar fallback - no 8-bit SIMD mul
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint8_t>(a[i] * b[i]);
    }
}

void Min(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinUint8)(out, a, b, count);
}

void Max(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxUint8)(out, a, b, count);
}

void Add(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddUint16)(out, a, b, count);
}

void Sub(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubUint16)(out, a, b, count);
}

void Mul(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count) {
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint16_t>(a[i] * b[i]);
    }
}

void Min(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinUint16)(out, a, b, count);
}

void Max(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, const uint16_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxUint16)(out, a, b, count);
}

void Add(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddUint32)(out, a, b, count);
}

void Sub(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubUint32)(out, a, b, count);
}

void Mul(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MulUint32)(out, a, b, count);
}

void Min(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MinUint32)(out, a, b, count);
}

void Max(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, const uint32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxUint32)(out, a, b, count);
}

void Add(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(AddUint64)(out, a, b, count);
}

void Sub(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(SubUint64)(out, a, b, count);
}

void Mul(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count) {
    // Scalar fallback - no 64-bit SIMD mul
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void Min(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count) {
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
    }
}

void Max(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, const uint64_t* HWY_RESTRICT b,
         size_t count) {
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

// =============================================================================
// Priority 3: Additional Math Operations
// =============================================================================

void Tan(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TanFloat32)(out, a, count);
}

void Tan(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TanFloat64)(out, a, count);
}

void Expm1(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Expm1Float32)(out, a, count);
}

void Expm1(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Expm1Float64)(out, a, count);
}

void Log1p(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Log1pFloat32)(out, a, count);
}

void Log1p(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Log1pFloat64)(out, a, count);
}

void AbsDiff(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
             size_t count) {
    HWY_DYNAMIC_DISPATCH(AbsDiffFloat32)(out, a, b, count);
}

void AbsDiff(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
             size_t count) {
    HWY_DYNAMIC_DISPATCH(AbsDiffFloat64)(out, a, b, count);
}

void AbsDiff(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
             const int32_t* HWY_RESTRICT b, size_t count) {
    // Scalar fallback for int32 absolute difference
    for (size_t i = 0; i < count; ++i) {
        int32_t diff = a[i] - b[i];
        out[i] = (diff >= 0) ? diff : -diff;
    }
}

void CopySign(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              size_t count) {
    HWY_DYNAMIC_DISPATCH(CopySignFloat32)(out, a, b, count);
}

void CopySign(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
              size_t count) {
    HWY_DYNAMIC_DISPATCH(CopySignFloat64)(out, a, b, count);
}

void ApproxReciprocal(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ApproxReciprocalFloat32)(out, a, count);
}

void ApproxReciprocalSqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ApproxReciprocalSqrtFloat32)(out, a, count);
}

void ApproxReciprocalSqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ApproxReciprocalSqrtFloat64)(out, a, count);
}

// =============================================================================
// Priority 3: Bit Counting
// =============================================================================

void PopCount(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PopCountInt32)(out, a, count);
}

void PopCount(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PopCountInt64)(out, a, count);
}

void PopCount(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PopCountUint32)(out, a, count);
}

void PopCount(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(PopCountUint64)(out, a, count);
}

void LeadingZeroCount(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(LeadingZeroCountInt32)(out, a, count);
}

void LeadingZeroCount(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(LeadingZeroCountInt64)(out, a, count);
}

void LeadingZeroCount(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(LeadingZeroCountUint32)(out, a, count);
}

void LeadingZeroCount(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(LeadingZeroCountUint64)(out, a, count);
}

void TrailingZeroCount(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TrailingZeroCountInt32)(out, a, count);
}

void TrailingZeroCount(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TrailingZeroCountInt64)(out, a, count);
}

void TrailingZeroCount(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TrailingZeroCountUint32)(out, a, count);
}

void TrailingZeroCount(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(TrailingZeroCountUint64)(out, a, count);
}

// =============================================================================
// Priority 3: Averaging
// =============================================================================

void AverageRound(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                  const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(AverageRoundUint8)(out, a, b, count);
}

void AverageRound(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                  const uint16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(AverageRoundUint16)(out, a, b, count);
}

// =============================================================================
// P0: Load/Store Operations
// =============================================================================

void Load(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadFloat32)(out, src, count);
}

void Load(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadFloat64)(out, src, count);
}

void Load(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInt32)(out, src, count);
}

void Load(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInt64)(out, src, count);
}

void Load(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadUint8)(out, src, count);
}

void Load(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadUint16)(out, src, count);
}

void Load(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadUint32)(out, src, count);
}

void Load(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadUint64)(out, src, count);
}

void Store(const float* HWY_RESTRICT src, float* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreFloat32)(src, dst, count);
}

void Store(const double* HWY_RESTRICT src, double* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreFloat64)(src, dst, count);
}

void Store(const int32_t* HWY_RESTRICT src, int32_t* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInt32)(src, dst, count);
}

void Store(const int64_t* HWY_RESTRICT src, int64_t* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInt64)(src, dst, count);
}

void Store(const uint8_t* HWY_RESTRICT src, uint8_t* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreUint8)(src, dst, count);
}

void Store(const uint16_t* HWY_RESTRICT src, uint16_t* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreUint16)(src, dst, count);
}

void Store(const uint32_t* HWY_RESTRICT src, uint32_t* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreUint32)(src, dst, count);
}

void Store(const uint64_t* HWY_RESTRICT src, uint64_t* HWY_RESTRICT dst, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreUint64)(src, dst, count);
}

// =============================================================================
// P0: BitCast Operations
// =============================================================================

void BitCastFloat32ToInt32(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastFloat32ToInt32Impl)(out, src, count);
}

void BitCastInt32ToFloat32(float* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastInt32ToFloat32Impl)(out, src, count);
}

void BitCastFloat64ToInt64(int64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                           size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastFloat64ToInt64Impl)(out, src, count);
}

void BitCastInt64ToFloat64(double* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                           size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastInt64ToFloat64Impl)(out, src, count);
}

void BitCastFloat32ToUint32(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                            size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastFloat32ToUint32Impl)(out, src, count);
}

void BitCastUint32ToFloat32(float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src,
                            size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastUint32ToFloat32Impl)(out, src, count);
}

void BitCastFloat64ToUint64(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                            size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastFloat64ToUint64Impl)(out, src, count);
}

void BitCastUint64ToFloat64(double* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src,
                            size_t count) {
    HWY_DYNAMIC_DISPATCH(BitCastUint64ToFloat64Impl)(out, src, count);
}

// =============================================================================
// P0: Mask Operations
// =============================================================================

size_t CountTrue(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CountTrueUint8Impl)(mask, count);
}

size_t CountTrue(const uint32_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CountTrueUint32Impl)(mask, count);
}

size_t CountTrue(const uint64_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CountTrueUint64Impl)(mask, count);
}

bool AllTrue(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(AllTrueUint8Impl)(mask, count);
}

bool AllTrue(const uint32_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(AllTrueUint32Impl)(mask, count);
}

bool AllTrue(const uint64_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(AllTrueUint64Impl)(mask, count);
}

bool AllFalse(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(AllFalseUint8Impl)(mask, count);
}

bool AllFalse(const uint32_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(AllFalseUint32Impl)(mask, count);
}

bool AllFalse(const uint64_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(AllFalseUint64Impl)(mask, count);
}

int64_t FindFirstTrue(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindFirstTrueUint8Impl)(mask, count);
}

int64_t FindFirstTrue(const uint32_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindFirstTrueUint32Impl)(mask, count);
}

int64_t FindFirstTrue(const uint64_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindFirstTrueUint64Impl)(mask, count);
}

int64_t FindLastTrue(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindLastTrueUint8Impl)(mask, count);
}

int64_t FindLastTrue(const uint32_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindLastTrueUint32Impl)(mask, count);
}

int64_t FindLastTrue(const uint64_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindLastTrueUint64Impl)(mask, count);
}

// =============================================================================
// P1: Reverse Operations
// =============================================================================

void Reverse(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseFloat32)(out, a, count);
}

void Reverse(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseFloat64)(out, a, count);
}

void Reverse(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseInt32)(out, a, count);
}

void Reverse(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseInt64)(out, a, count);
}

void Reverse(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseUint8)(out, a, count);
}

void Reverse(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseUint16)(out, a, count);
}

void Reverse(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseUint32)(out, a, count);
}

void Reverse(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseUint64)(out, a, count);
}

// =============================================================================
// P1: Fill Operations
// =============================================================================

void Fill(float* HWY_RESTRICT data, float value, size_t count) {
    HWY_DYNAMIC_DISPATCH(FillFloat32)(data, value, count);
}

void Fill(double* HWY_RESTRICT data, double value, size_t count) {
    HWY_DYNAMIC_DISPATCH(FillFloat64)(data, value, count);
}

void Fill(int32_t* HWY_RESTRICT data, int32_t value, size_t count) {
    HWY_DYNAMIC_DISPATCH(FillInt32)(data, value, count);
}

void Fill(int64_t* HWY_RESTRICT data, int64_t value, size_t count) {
    HWY_DYNAMIC_DISPATCH(FillInt64)(data, value, count);
}

void Fill(uint8_t* HWY_RESTRICT data, uint8_t value, size_t count) {
    HWY_DYNAMIC_DISPATCH(FillUint8)(data, value, count);
}

// =============================================================================
// P1: Copy Operations
// =============================================================================

void Copy(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(CopyFloat32)(dst, src, count);
}

void Copy(double* HWY_RESTRICT dst, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(CopyFloat64)(dst, src, count);
}

void Copy(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(CopyInt32)(dst, src, count);
}

void Copy(int64_t* HWY_RESTRICT dst, const int64_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(CopyInt64)(dst, src, count);
}

void Copy(uint8_t* HWY_RESTRICT dst, const uint8_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(CopyUint8)(dst, src, count);
}

// =============================================================================
// P1: MinOfLanes/MaxOfLanes Operations
// =============================================================================

void MinOfLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MinOfLanesFloat32)(out, a, count);
}

void MinOfLanes(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MinOfLanesFloat64)(out, a, count);
}

void MinOfLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MinOfLanesInt32)(out, a, count);
}

void MinOfLanes(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MinOfLanesInt64)(out, a, count);
}

void MinOfLanes(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MinOfLanesUint32)(out, a, count);
}

void MaxOfLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxOfLanesFloat32)(out, a, count);
}

void MaxOfLanes(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxOfLanesFloat64)(out, a, count);
}

void MaxOfLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxOfLanesInt32)(out, a, count);
}

void MaxOfLanes(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxOfLanesInt64)(out, a, count);
}

void MaxOfLanes(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxOfLanesUint32)(out, a, count);
}

// =============================================================================
// P1: Interleaved Load/Store Operations
// =============================================================================

void LoadInterleaved2(float* HWY_RESTRICT a, float* HWY_RESTRICT b,
                      const float* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved2Float32)(a, b, interleaved, count);
}

void LoadInterleaved2(int32_t* HWY_RESTRICT a, int32_t* HWY_RESTRICT b,
                      const int32_t* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved2Int32)(a, b, interleaved, count);
}

void LoadInterleaved2(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b,
                      const uint8_t* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved2Uint8)(a, b, interleaved, count);
}

void LoadInterleaved3(float* HWY_RESTRICT a, float* HWY_RESTRICT b, float* HWY_RESTRICT c,
                      const float* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved3Float32)(a, b, c, interleaved, count);
}

void LoadInterleaved3(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b, uint8_t* HWY_RESTRICT c,
                      const uint8_t* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved3Uint8)(a, b, c, interleaved, count);
}

void LoadInterleaved4(float* HWY_RESTRICT a, float* HWY_RESTRICT b, float* HWY_RESTRICT c,
                      float* HWY_RESTRICT d, const float* HWY_RESTRICT interleaved, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved4Float32)(a, b, c, d, interleaved, count);
}

void LoadInterleaved4(uint8_t* HWY_RESTRICT a, uint8_t* HWY_RESTRICT b, uint8_t* HWY_RESTRICT c,
                      uint8_t* HWY_RESTRICT d, const uint8_t* HWY_RESTRICT interleaved,
                      size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadInterleaved4Uint8)(a, b, c, d, interleaved, count);
}

void StoreInterleaved2(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved2Float32)(interleaved, a, b, count);
}

void StoreInterleaved2(int32_t* HWY_RESTRICT interleaved, const int32_t* HWY_RESTRICT a,
                       const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved2Int32)(interleaved, a, b, count);
}

void StoreInterleaved2(uint8_t* HWY_RESTRICT interleaved, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved2Uint8)(interleaved, a, b, count);
}

void StoreInterleaved3(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved3Float32)(interleaved, a, b, c, count);
}

void StoreInterleaved3(uint8_t* HWY_RESTRICT interleaved, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved3Uint8)(interleaved, a, b, c, count);
}

void StoreInterleaved4(float* HWY_RESTRICT interleaved, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c,
                       const float* HWY_RESTRICT d, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved4Float32)(interleaved, a, b, c, d, count);
}

void StoreInterleaved4(uint8_t* HWY_RESTRICT interleaved, const uint8_t* HWY_RESTRICT a,
                       const uint8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT c,
                       const uint8_t* HWY_RESTRICT d, size_t count) {
    HWY_DYNAMIC_DISPATCH(StoreInterleaved4Uint8)(interleaved, a, b, c, d, count);
}

// =============================================================================
// P1: Find Operations
// =============================================================================

int64_t Find(const float* HWY_RESTRICT data, float value, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindFloat32)(data, value, count);
}

int64_t Find(const int32_t* HWY_RESTRICT data, int32_t value, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindInt32)(data, value, count);
}

int64_t Find(const uint8_t* HWY_RESTRICT data, uint8_t value, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindUint8)(data, value, count);
}

int64_t FindGt(const float* HWY_RESTRICT data, float threshold, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindGtFloat32)(data, threshold, count);
}

int64_t FindGt(const int32_t* HWY_RESTRICT data, int32_t threshold, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindGtInt32)(data, threshold, count);
}

int64_t FindLt(const float* HWY_RESTRICT data, float threshold, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindLtFloat32)(data, threshold, count);
}

int64_t FindLt(const int32_t* HWY_RESTRICT data, int32_t threshold, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindLtInt32)(data, threshold, count);
}

// =============================================================================
// P1: Transform Operations
// =============================================================================

void TransformAdd(float* HWY_RESTRICT data, float scalar, size_t count) {
    HWY_DYNAMIC_DISPATCH(TransformAddFloat32)(data, scalar, count);
}

void TransformAdd(int32_t* HWY_RESTRICT data, int32_t scalar, size_t count) {
    HWY_DYNAMIC_DISPATCH(TransformAddInt32)(data, scalar, count);
}

void TransformMul(float* HWY_RESTRICT data, float scalar, size_t count) {
    HWY_DYNAMIC_DISPATCH(TransformMulFloat32)(data, scalar, count);
}

void TransformMul(int32_t* HWY_RESTRICT data, int32_t scalar, size_t count) {
    HWY_DYNAMIC_DISPATCH(TransformMulInt32)(data, scalar, count);
}

// =============================================================================
// P1: VQSort Integration (Pre-compiled library, no HWY_EXPORT needed)
// =============================================================================

void Sort(float* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortAscending());
}

void Sort(double* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortAscending());
}

void Sort(int32_t* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortAscending());
}

void Sort(int64_t* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortAscending());
}

void Sort(uint32_t* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortAscending());
}

void Sort(uint64_t* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortAscending());
}

void SortDescending(float* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortDescending());
}

void SortDescending(double* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortDescending());
}

void SortDescending(int32_t* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortDescending());
}

void SortDescending(int64_t* HWY_RESTRICT data, size_t count) {
    if (count <= 1)
        return;
    hwy::VQSort(data, count, hwy::SortDescending());
}

void PartialSort(float* HWY_RESTRICT data, size_t k, size_t count) {
    if (count <= 1 || k == 0)
        return;
    if (k >= count) {
        hwy::VQSort(data, count, hwy::SortAscending());
    } else {
        hwy::VQPartialSort(data, count, k, hwy::SortAscending());
    }
}

void PartialSort(int32_t* HWY_RESTRICT data, size_t k, size_t count) {
    if (count <= 1 || k == 0)
        return;
    if (k >= count) {
        hwy::VQSort(data, count, hwy::SortAscending());
    } else {
        hwy::VQPartialSort(data, count, k, hwy::SortAscending());
    }
}

// =============================================================================
// P2: SinCos (Combined sin and cos for efficiency)
// =============================================================================

void SinCos(float* HWY_RESTRICT sin_out, float* HWY_RESTRICT cos_out, const float* HWY_RESTRICT a,
            size_t count) {
    // Use separate Sin and Cos calls - Highway's SinCos may not be available
    // in all contrib versions. This is still faster than scalar due to SIMD.
    HWY_DYNAMIC_DISPATCH(SinFloat32)(sin_out, a, count);
    HWY_DYNAMIC_DISPATCH(CosFloat32)(cos_out, a, count);
}

void SinCos(double* HWY_RESTRICT sin_out, double* HWY_RESTRICT cos_out,
            const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SinFloat64)(sin_out, a, count);
    HWY_DYNAMIC_DISPATCH(CosFloat64)(cos_out, a, count);
}

// =============================================================================
// P2: Hypot (sqrt(a^2 + b^2))
// =============================================================================

void Hypot(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(HypotFloat32)(out, a, b, count);
}

void Hypot(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
           size_t count) {
    HWY_DYNAMIC_DISPATCH(HypotFloat64)(out, a, b, count);
}

// =============================================================================
// P2: MatVec (Matrix-vector multiplication)
// =============================================================================

void MatVec(float* HWY_RESTRICT out, const float* HWY_RESTRICT A, const float* HWY_RESTRICT x,
            size_t rows, size_t cols) {
    HWY_DYNAMIC_DISPATCH(MatVecFloat32)(out, A, x, rows, cols);
}

void MatVec(double* HWY_RESTRICT out, const double* HWY_RESTRICT A, const double* HWY_RESTRICT x,
            size_t rows, size_t cols) {
    HWY_DYNAMIC_DISPATCH(MatVecFloat64)(out, A, x, rows, cols);
}

// =============================================================================
// P2: Pow (Power function)
// =============================================================================

void Pow(float* HWY_RESTRICT out, const float* HWY_RESTRICT base, const float* HWY_RESTRICT exp,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(PowFloat32)(out, base, exp, count);
}

void Pow(double* HWY_RESTRICT out, const double* HWY_RESTRICT base, const double* HWY_RESTRICT exp,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(PowFloat64)(out, base, exp, count);
}

void PowScalar(float* HWY_RESTRICT out, const float* HWY_RESTRICT base, float scalar_exp,
               size_t count) {
    HWY_DYNAMIC_DISPATCH(PowScalarFloat32)(out, base, scalar_exp, count);
}

void PowScalar(double* HWY_RESTRICT out, const double* HWY_RESTRICT base, double scalar_exp,
               size_t count) {
    HWY_DYNAMIC_DISPATCH(PowScalarFloat64)(out, base, scalar_exp, count);
}

// =============================================================================
// P2: Half-Precision Conversions (Float16 / BFloat16)
// =============================================================================
// These use Highway's scalar conversion functions which are highly optimized.
// For bulk conversions, we unroll the loop for better pipelining.

void F32ToF16(uint16_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    size_t i = 0;
    // Unroll by 4 for better instruction-level parallelism
    for (; i + 4 <= count; i += 4) {
        const hwy::float16_t f0 = hwy::F16FromF32(in[i + 0]);
        const hwy::float16_t f1 = hwy::F16FromF32(in[i + 1]);
        const hwy::float16_t f2 = hwy::F16FromF32(in[i + 2]);
        const hwy::float16_t f3 = hwy::F16FromF32(in[i + 3]);
        out[i + 0] = hwy::BitCastScalar<uint16_t>(f0);
        out[i + 1] = hwy::BitCastScalar<uint16_t>(f1);
        out[i + 2] = hwy::BitCastScalar<uint16_t>(f2);
        out[i + 3] = hwy::BitCastScalar<uint16_t>(f3);
    }
    // Handle remainder
    for (; i < count; ++i) {
        out[i] = hwy::BitCastScalar<uint16_t>(hwy::F16FromF32(in[i]));
    }
}

void F16ToF32(float* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT in, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        out[i + 0] = hwy::F32FromF16(hwy::float16_t::FromBits(in[i + 0]));
        out[i + 1] = hwy::F32FromF16(hwy::float16_t::FromBits(in[i + 1]));
        out[i + 2] = hwy::F32FromF16(hwy::float16_t::FromBits(in[i + 2]));
        out[i + 3] = hwy::F32FromF16(hwy::float16_t::FromBits(in[i + 3]));
    }
    for (; i < count; ++i) {
        out[i] = hwy::F32FromF16(hwy::float16_t::FromBits(in[i]));
    }
}

void F32ToBF16(uint16_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const hwy::bfloat16_t bf0 = hwy::BF16FromF32(in[i + 0]);
        const hwy::bfloat16_t bf1 = hwy::BF16FromF32(in[i + 1]);
        const hwy::bfloat16_t bf2 = hwy::BF16FromF32(in[i + 2]);
        const hwy::bfloat16_t bf3 = hwy::BF16FromF32(in[i + 3]);
        out[i + 0] = hwy::BitCastScalar<uint16_t>(bf0);
        out[i + 1] = hwy::BitCastScalar<uint16_t>(bf1);
        out[i + 2] = hwy::BitCastScalar<uint16_t>(bf2);
        out[i + 3] = hwy::BitCastScalar<uint16_t>(bf3);
    }
    for (; i < count; ++i) {
        out[i] = hwy::BitCastScalar<uint16_t>(hwy::BF16FromF32(in[i]));
    }
}

void BF16ToF32(float* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT in, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        out[i + 0] = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(in[i + 0]));
        out[i + 1] = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(in[i + 1]));
        out[i + 2] = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(in[i + 2]));
        out[i + 3] = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(in[i + 3]));
    }
    for (; i < count; ++i) {
        out[i] = hwy::F32FromBF16(hwy::bfloat16_t::FromBits(in[i]));
    }
}

void F64ToF16(uint16_t* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const hwy::float16_t f0 = hwy::F16FromF64(in[i + 0]);
        const hwy::float16_t f1 = hwy::F16FromF64(in[i + 1]);
        const hwy::float16_t f2 = hwy::F16FromF64(in[i + 2]);
        const hwy::float16_t f3 = hwy::F16FromF64(in[i + 3]);
        out[i + 0] = hwy::BitCastScalar<uint16_t>(f0);
        out[i + 1] = hwy::BitCastScalar<uint16_t>(f1);
        out[i + 2] = hwy::BitCastScalar<uint16_t>(f2);
        out[i + 3] = hwy::BitCastScalar<uint16_t>(f3);
    }
    for (; i < count; ++i) {
        out[i] = hwy::BitCastScalar<uint16_t>(hwy::F16FromF64(in[i]));
    }
}

void F64ToBF16(uint16_t* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const hwy::bfloat16_t bf0 = hwy::BF16FromF64(in[i + 0]);
        const hwy::bfloat16_t bf1 = hwy::BF16FromF64(in[i + 1]);
        const hwy::bfloat16_t bf2 = hwy::BF16FromF64(in[i + 2]);
        const hwy::bfloat16_t bf3 = hwy::BF16FromF64(in[i + 3]);
        out[i + 0] = hwy::BitCastScalar<uint16_t>(bf0);
        out[i + 1] = hwy::BitCastScalar<uint16_t>(bf1);
        out[i + 2] = hwy::BitCastScalar<uint16_t>(bf2);
        out[i + 3] = hwy::BitCastScalar<uint16_t>(bf3);
    }
    for (; i < count; ++i) {
        out[i] = hwy::BitCastScalar<uint16_t>(hwy::BF16FromF64(in[i]));
    }
}

// =============================================================================
// P0: Masked Arithmetic Operations
// =============================================================================

void MaskedAdd(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAddFloat32)(out, mask, a, b, no, count);
}

void MaskedAdd(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAddFloat64)(out, mask, a, b, no, count);
}

void MaskedAdd(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAddInt32)(out, mask, a, b, no, count);
}

void MaskedAdd(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
               const int64_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAddInt64)(out, mask, a, b, no, count);
}

void MaskedSub(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSubFloat32)(out, mask, a, b, no, count);
}

void MaskedSub(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSubFloat64)(out, mask, a, b, no, count);
}

void MaskedSub(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSubInt32)(out, mask, a, b, no, count);
}

void MaskedSub(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
               const int64_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSubInt64)(out, mask, a, b, no, count);
}

void MaskedMul(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMulFloat32)(out, mask, a, b, no, count);
}

void MaskedMul(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMulFloat64)(out, mask, a, b, no, count);
}

void MaskedMul(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMulInt32)(out, mask, a, b, no, count);
}

void MaskedMul(int64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
               const int64_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMulInt64)(out, mask, a, b, no, count);
}

void MaskedDiv(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedDivFloat32)(out, mask, a, b, no, count);
}

void MaskedDiv(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedDivFloat64)(out, mask, a, b, no, count);
}

void MaskedMin(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMinFloat32)(out, mask, a, b, no, count);
}

void MaskedMin(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMinFloat64)(out, mask, a, b, no, count);
}

void MaskedMin(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMinInt32)(out, mask, a, b, no, count);
}

void MaskedMax(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMaxFloat32)(out, mask, a, b, no, count);
}

void MaskedMax(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMaxFloat64)(out, mask, a, b, no, count);
}

void MaskedMax(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
               const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMaxInt32)(out, mask, a, b, no, count);
}

void MaskedAbs(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAbsFloat32)(out, mask, a, no, count);
}

void MaskedAbs(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAbsFloat64)(out, mask, a, no, count);
}

void MaskedAbs(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAbsInt32)(out, mask, a, no, count);
}

void MaskedNeg(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const float* HWY_RESTRICT a, const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedNegFloat32)(out, mask, a, no, count);
}

void MaskedNeg(double* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const double* HWY_RESTRICT a, const double* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedNegFloat64)(out, mask, a, no, count);
}

void MaskedNeg(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
               const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedNegInt32)(out, mask, a, no, count);
}

// =============================================================================
// P0: Widening Operations
// =============================================================================

void SumsOf2(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOf2Int16)(out, a, count);
}

void SumsOf2(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOf2Uint16)(out, a, count);
}

void SumsOf4(int32_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOf4Int8)(out, a, count);
}

void SumsOf4(uint32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOf4Uint8)(out, a, count);
}

void MulEven(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
             const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulEvenInt32)(out, a, b, count);
}

void MulEven(uint64_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
             const uint32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulEvenUint32)(out, a, b, count);
}

void MulOdd(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
            size_t count) {
    HWY_DYNAMIC_DISPATCH(MulOddInt32)(out, a, b, count);
}

void MulOdd(uint64_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a,
            const uint32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulOddUint32)(out, a, b, count);
}

// =============================================================================
// P0: Additional Comparison Operations
// =============================================================================

void IsNegative(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsNegativeFloat32)(out, a, count);
}

void IsNegative(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsNegativeFloat64)(out, a, count);
}

void IsNegative(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsNegativeInt32)(out, a, count);
}

void IsEitherNaN(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                 const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsEitherNaNFloat32)(out, a, b, count);
}

void IsEitherNaN(uint8_t* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                 const double* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(IsEitherNaNFloat64)(out, a, b, count);
}

// =============================================================================
// P1: Extended FMA Operations
// =============================================================================

void NegMulSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegMulSubFloat32)(out, a, b, c, count);
}

void NegMulSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(NegMulSubFloat64)(out, a, b, c, count);
}

void MulAddSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulAddSubFloat32)(out, a, b, c, count);
}

void MulAddSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulAddSubFloat64)(out, a, b, c, count);
}

// =============================================================================
// P1: CompressStore Operations
// =============================================================================

size_t CompressStore(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                     const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressStoreFloat32)(out, a, mask, count);
}

size_t CompressStore(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                     const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressStoreFloat64)(out, a, mask, count);
}

size_t CompressStore(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                     const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressStoreInt32)(out, a, mask, count);
}

// =============================================================================
// P2: Iota and FirstN Operations
// =============================================================================

void Iota(float* HWY_RESTRICT out, float start, size_t count) {
    HWY_DYNAMIC_DISPATCH(IotaFloat32)(out, start, count);
}

void Iota(double* HWY_RESTRICT out, double start, size_t count) {
    HWY_DYNAMIC_DISPATCH(IotaFloat64)(out, start, count);
}

void Iota(int32_t* HWY_RESTRICT out, int32_t start, size_t count) {
    HWY_DYNAMIC_DISPATCH(IotaInt32)(out, start, count);
}

void Iota(int64_t* HWY_RESTRICT out, int64_t start, size_t count) {
    HWY_DYNAMIC_DISPATCH(IotaInt64)(out, start, count);
}

void Iota(uint32_t* HWY_RESTRICT out, uint32_t start, size_t count) {
    HWY_DYNAMIC_DISPATCH(IotaUint32)(out, start, count);
}

void Iota(uint64_t* HWY_RESTRICT out, uint64_t start, size_t count) {
    HWY_DYNAMIC_DISPATCH(IotaUint64)(out, start, count);
}

void FirstN(uint8_t* HWY_RESTRICT out, size_t n, size_t count) {
    HWY_DYNAMIC_DISPATCH(FirstNImpl)(out, n, count);
}

// =============================================================================
// P0: WidenMulAccumulate Operations
// =============================================================================

void WidenMulAccumulate(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                        const int16_t* HWY_RESTRICT b, const int32_t* HWY_RESTRICT c,
                        size_t count) {
    HWY_DYNAMIC_DISPATCH(WidenMulAccumulateInt16)(out, a, b, c, count);
}

void WidenMulAccumulate(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                        const uint16_t* HWY_RESTRICT b, const uint32_t* HWY_RESTRICT c,
                        size_t count) {
    HWY_DYNAMIC_DISPATCH(WidenMulAccumulateUint16)(out, a, b, c, count);
}

// =============================================================================
// P0: SumsOf8 Operation
// =============================================================================

void SumsOf8(uint64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOf8Uint8)(out, a, count);
}

// =============================================================================
// P0: TestBit Operation
// =============================================================================

void TestBit(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t bit, size_t count) {
    HWY_DYNAMIC_DISPATCH(TestBitInt32)(out, a, bit, count);
}

void TestBit(uint8_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, size_t bit, size_t count) {
    HWY_DYNAMIC_DISPATCH(TestBitInt64)(out, a, bit, count);
}

void TestBit(uint8_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t bit, size_t count) {
    HWY_DYNAMIC_DISPATCH(TestBitUint32)(out, a, bit, count);
}

void TestBit(uint8_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t bit, size_t count) {
    HWY_DYNAMIC_DISPATCH(TestBitUint64)(out, a, bit, count);
}

// =============================================================================
// P1: MulSubAdd Operations
// =============================================================================

void MulSubAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulSubAddFloat32)(out, a, b, c, count);
}

void MulSubAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
               const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulSubAddFloat64)(out, a, b, c, count);
}

// =============================================================================
// P1: Reverse2, Reverse4, Reverse8 Operations
// =============================================================================

void Reverse2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Reverse2Float32)(out, a, count);
}

void Reverse2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Reverse2Float64)(out, a, count);
}

void Reverse2(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Reverse2Int32)(out, a, count);
}

void Reverse4(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Reverse4Float32)(out, a, count);
}

void Reverse4(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Reverse4Int32)(out, a, count);
}

void Reverse8(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Reverse8Uint8)(out, a, count);
}

// =============================================================================
// P1: DupEven, DupOdd Operations
// =============================================================================

void DupEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(DupEvenFloat32)(out, a, count);
}

void DupEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(DupEvenInt32)(out, a, count);
}

void DupOdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(DupOddFloat32)(out, a, count);
}

void DupOdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(DupOddInt32)(out, a, count);
}

// =============================================================================
// P1: InterleaveLower, InterleaveUpper Operations
// =============================================================================

void InterleaveLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                     const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveLowerFloat32)(out, a, b, count);
}

void InterleaveLower(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                     const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveLowerInt32)(out, a, b, count);
}

void InterleaveUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                     const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveUpperFloat32)(out, a, b, count);
}

void InterleaveUpper(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                     const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveUpperInt32)(out, a, b, count);
}

// =============================================================================
// P1: Mask Logical Operations
// =============================================================================

void MaskNot(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskNotUint8)(out, mask, count);
}

void MaskAnd(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
             const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskAndUint8)(out, a, b, count);
}

void MaskOr(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a, const uint8_t* HWY_RESTRICT b,
            size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskOrUint8)(out, a, b, count);
}

void MaskXor(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
             const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskXorUint8)(out, a, b, count);
}

void MaskAndNot(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskAndNotUint8)(out, a, b, count);
}

// =============================================================================
// P2: AddSub Operations
// =============================================================================

void AddSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            size_t count) {
    HWY_DYNAMIC_DISPATCH(AddSubFloat32)(out, a, b, count);
}

void AddSub(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, const double* HWY_RESTRICT b,
            size_t count) {
    HWY_DYNAMIC_DISPATCH(AddSubFloat64)(out, a, b, count);
}

// =============================================================================
// P2: MinMagnitude, MaxMagnitude Operations
// =============================================================================

void MinMagnitude(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                  size_t count) {
    HWY_DYNAMIC_DISPATCH(MinMagnitudeFloat32)(out, a, b, count);
}

void MinMagnitude(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                  const double* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MinMagnitudeFloat64)(out, a, b, count);
}

void MaxMagnitude(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                  size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxMagnitudeFloat32)(out, a, b, count);
}

void MaxMagnitude(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                  const double* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaxMagnitudeFloat64)(out, a, b, count);
}

// =============================================================================
// P2: MaskedLoad, MaskedStore, BlendedStore Operations
// =============================================================================

void MaskedLoad(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, float fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLoadFloat32)(out, src, mask, fallback, count);
}

void MaskedLoad(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, double fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLoadFloat64)(out, src, mask, fallback, count);
}

void MaskedLoad(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, int32_t fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLoadInt32)(out, src, mask, fallback, count);
}

void MaskedStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                 const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedStoreFloat32)(dst, src, mask, count);
}

void MaskedStore(double* HWY_RESTRICT dst, const double* HWY_RESTRICT src,
                 const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedStoreFloat64)(dst, src, mask, count);
}

void MaskedStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                 const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedStoreInt32)(dst, src, mask, count);
}

void BlendedStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT new_val,
                  const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(BlendedStoreFloat32)(dst, new_val, mask, count);
}

void BlendedStore(double* HWY_RESTRICT dst, const double* HWY_RESTRICT new_val,
                  const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(BlendedStoreFloat64)(dst, new_val, mask, count);
}

void BlendedStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT new_val,
                  const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(BlendedStoreInt32)(dst, new_val, mask, count);
}

// =============================================================================
// P0: WidenMulPairwiseAdd Operations
// =============================================================================

void WidenMulPairwiseAdd(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                         const int16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(WidenMulPairwiseAddInt16)(out, a, b, count);
}

void WidenMulPairwiseAdd(uint32_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                         const uint16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(WidenMulPairwiseAddUint16)(out, a, b, count);
}

// =============================================================================
// P1: BroadcastLane Operations
// =============================================================================

void BroadcastLane(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t lane,
                   size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastLaneFloat32)(out, a, lane, count);
}

void BroadcastLane(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t lane,
                   size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastLaneFloat64)(out, a, lane, count);
}

void BroadcastLane(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t lane,
                   size_t count) {
    HWY_DYNAMIC_DISPATCH(BroadcastLaneInt32)(out, a, lane, count);
}

// =============================================================================
// P1: Slide Operations
// =============================================================================

void Slide1Up(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Slide1UpFloat32)(out, a, count);
}

void Slide1Up(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Slide1UpInt32)(out, a, count);
}

void Slide1Down(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Slide1DownFloat32)(out, a, count);
}

void Slide1Down(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(Slide1DownInt32)(out, a, count);
}

// =============================================================================
// P1: Concat Operations
// =============================================================================

void ConcatLowerUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                      const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatLowerUpperFloat32)(out, a, b, count);
}

void ConcatLowerUpper(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatLowerUpperInt32)(out, a, b, count);
}

void ConcatUpperLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                      const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatUpperLowerFloat32)(out, a, b, count);
}

void ConcatUpperLower(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                      const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatUpperLowerInt32)(out, a, b, count);
}

void ConcatEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatEvenFloat32)(out, a, b, count);
}

void ConcatEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatEvenInt32)(out, a, b, count);
}

void ConcatOdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
               size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatOddFloat32)(out, a, b, count);
}

void ConcatOdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConcatOddInt32)(out, a, b, count);
}

// =============================================================================
// P1: Mask Utility Operations
// =============================================================================

size_t FindKnownFirstTrue(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindKnownFirstTrueImpl)(mask, count);
}

size_t FindKnownLastTrue(const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindKnownLastTrueImpl)(mask, count);
}

size_t StoreMaskBits(uint8_t* HWY_RESTRICT bits_out, const uint8_t* HWY_RESTRICT mask,
                     size_t count) {
    return HWY_DYNAMIC_DISPATCH(StoreMaskBitsImpl)(bits_out, mask, count);
}

void LoadMaskBits(uint8_t* HWY_RESTRICT mask_out, const uint8_t* HWY_RESTRICT bits, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadMaskBitsImpl)(mask_out, bits, count);
}

// =============================================================================
// P1: CompressBlendedStore and CompressNot Operations
// =============================================================================

size_t CompressBlendedStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                            const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressBlendedStoreFloat32)(dst, src, mask, count);
}

size_t CompressBlendedStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                            const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressBlendedStoreInt32)(dst, src, mask, count);
}

size_t CompressNot(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                   const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressNotFloat32)(out, src, mask, count);
}

size_t CompressNot(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                   const uint8_t* HWY_RESTRICT mask, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressNotInt32)(out, src, mask, count);
}

// =============================================================================
// P1: InterleaveEven and InterleaveOdd Operations
// =============================================================================

void InterleaveEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                    const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveEvenFloat32)(out, a, b, count);
}

void InterleaveEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                    const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveEvenInt32)(out, a, b, count);
}

void InterleaveOdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                   const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveOddFloat32)(out, a, b, count);
}

void InterleaveOdd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                   const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveOddInt32)(out, a, b, count);
}

// =============================================================================
// P1: Shuffle Operations
// =============================================================================

void Shuffle0123(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle0123Float32)(out, in, count);
}

void Shuffle0123(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle0123Int32)(out, in, count);
}

void Shuffle2301(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle2301Float32)(out, in, count);
}

void Shuffle2301(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle2301Int32)(out, in, count);
}

void Shuffle1032(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle1032Float32)(out, in, count);
}

void Shuffle1032(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle1032Int32)(out, in, count);
}

void Shuffle01(double* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle01Float64)(out, in, count);
}

void Shuffle01(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle01Int64)(out, in, count);
}

void Shuffle10(double* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle10Float64)(out, in, count);
}

void Shuffle10(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(Shuffle10Int64)(out, in, count);
}

// =============================================================================
// P1: TableLookup Operations
// =============================================================================

void TableLookupBytes(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT table,
                      const uint8_t* HWY_RESTRICT indices, size_t count, size_t table_size) {
    HWY_DYNAMIC_DISPATCH(TableLookupBytesUint8)(out, table, indices, count, table_size);
}

void TableLookupLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT table,
                      const int32_t* HWY_RESTRICT indices, size_t count, size_t table_size) {
    HWY_DYNAMIC_DISPATCH(TableLookupLanesFloat32)(out, table, indices, count, table_size);
}

void TableLookupLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT table,
                      const int32_t* HWY_RESTRICT indices, size_t count, size_t table_size) {
    HWY_DYNAMIC_DISPATCH(TableLookupLanesInt32)(out, table, indices, count, table_size);
}

// =============================================================================
// P1: Mask Set Operations
// =============================================================================

void SetBeforeFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(SetBeforeFirstImpl)(out, mask, count);
}

void SetAtOrBeforeFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(SetAtOrBeforeFirstImpl)(out, mask, count);
}

void SetOnlyFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(SetOnlyFirstImpl)(out, mask, count);
}

void SetAtOrAfterFirst(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(SetAtOrAfterFirstImpl)(out, mask, count);
}

// =============================================================================
// P1: Masked Reduction Operations
// =============================================================================

float MaskedReduceSum(const float* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                      size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceSumFloat32)(src, mask, count);
}

double MaskedReduceSum(const double* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                       size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceSumFloat64)(src, mask, count);
}

int32_t MaskedReduceSum(const int32_t* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                        size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceSumInt32)(src, mask, count);
}

float MaskedReduceMin(const float* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                      size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceMinFloat32)(src, mask, count);
}

double MaskedReduceMin(const double* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                       size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceMinFloat64)(src, mask, count);
}

int32_t MaskedReduceMin(const int32_t* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                        size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceMinInt32)(src, mask, count);
}

float MaskedReduceMax(const float* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                      size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceMaxFloat32)(src, mask, count);
}

double MaskedReduceMax(const double* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                       size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceMaxFloat64)(src, mask, count);
}

int32_t MaskedReduceMax(const int32_t* HWY_RESTRICT src, const uint8_t* HWY_RESTRICT mask,
                        size_t count) {
    return HWY_DYNAMIC_DISPATCH(MaskedReduceMaxInt32)(src, mask, count);
}

// =============================================================================
// Remaining P1 Operations
// =============================================================================

void TwoTablesLookupLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT table0,
                          const float* HWY_RESTRICT table1, const int32_t* HWY_RESTRICT indices,
                          size_t count, size_t table_size) {
    HWY_DYNAMIC_DISPATCH(TwoTablesLookupLanesFloat32)
    (out, table0, table1, indices, count, table_size);
}

void TwoTablesLookupLanes(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT table0,
                          const int32_t* HWY_RESTRICT table1, const int32_t* HWY_RESTRICT indices,
                          size_t count, size_t table_size) {
    HWY_DYNAMIC_DISPATCH(TwoTablesLookupLanesInt32)
    (out, table0, table1, indices, count, table_size);
}

size_t CompressBits(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                    const uint8_t* HWY_RESTRICT bits, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressBitsFloat32)(out, src, bits, count);
}

size_t CompressBits(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                    const uint8_t* HWY_RESTRICT bits, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressBitsInt32)(out, src, bits, count);
}

size_t CompressBitsStore(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                         const uint8_t* HWY_RESTRICT bits, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressBitsStoreFloat32)(dst, src, bits, count);
}

size_t CompressBitsStore(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src,
                         const uint8_t* HWY_RESTRICT bits, size_t count) {
    return HWY_DYNAMIC_DISPATCH(CompressBitsStoreInt32)(dst, src, bits, count);
}

void LoadExpand(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadExpandFloat32)(out, src, mask, count);
}

void LoadExpand(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadExpandInt32)(out, src, mask, count);
}

void PairwiseSub(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                 size_t count) {
    HWY_DYNAMIC_DISPATCH(PairwiseSubFloat32)(out, a, b, count);
}

void PairwiseSub(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
                 const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(PairwiseSubInt32)(out, a, b, count);
}

void SumsOfAdjQuadAbsDiff(uint16_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                          const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOfAdjQuadAbsDiffUint8)(out, a, b, count);
}

// =============================================================================
// P2: Math Functions
// =============================================================================

void Cbrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CbrtFloat32)(out, a, count);
}

void Cbrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(CbrtFloat64)(out, a, count);
}

void Erf(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ErfFloat32)(out, a, count);
}

void Erf(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ErfFloat64)(out, a, count);
}

void Erfc(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ErfcFloat32)(out, a, count);
}

void Erfc(double* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ErfcFloat64)(out, a, count);
}

// =============================================================================
// P2: Generation Operations
// =============================================================================

void IndicesFromVec(int32_t* HWY_RESTRICT indices_out, const uint8_t* HWY_RESTRICT mask,
                    size_t count) {
    HWY_DYNAMIC_DISPATCH(IndicesFromVecImpl)(indices_out, mask, count);
}

void IndicesFromNotVec(int32_t* HWY_RESTRICT indices_out, const uint8_t* HWY_RESTRICT mask,
                       size_t count) {
    HWY_DYNAMIC_DISPATCH(IndicesFromNotVecImpl)(indices_out, mask, count);
}

// =============================================================================
// P2: Type Conversions
// =============================================================================

void PromoteLowerTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteLowerToFloat64)(out, in, count);
}

void PromoteLowerTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteLowerToInt32)(out, in, count);
}

void PromoteLowerTo(int64_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteLowerToInt64)(out, in, count);
}

void PromoteUpperTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count,
                    size_t half) {
    HWY_DYNAMIC_DISPATCH(PromoteUpperToFloat64)(out, in, count, half);
}

void PromoteUpperTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count,
                    size_t half) {
    HWY_DYNAMIC_DISPATCH(PromoteUpperToInt32)(out, in, count, half);
}

void PromoteEvenTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteEvenToFloat64)(out, in, count);
}

void PromoteEvenTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteEvenToInt32)(out, in, count);
}

void PromoteOddTo(double* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteOddToFloat64)(out, in, count);
}

void PromoteOddTo(int32_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteOddToInt32)(out, in, count);
}

// =============================================================================
// P2: Additional Arithmetic Operations
// =============================================================================

void Mod(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(ModInt32)(out, a, b, count);
}

void Mod(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         size_t count) {
    HWY_DYNAMIC_DISPATCH(ModInt64)(out, a, b, count);
}

void SaturatedNeg(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedNegInt8)(out, a, count);
}

void SaturatedNeg(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedNegInt16)(out, a, count);
}

void SaturatedAbs(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedAbsInt8)(out, a, count);
}

void SaturatedAbs(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedAbsInt16)(out, a, count);
}

// =============================================================================
// P2: Bitwise Operations
// =============================================================================

void Xor3(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
          const int32_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(Xor3Int32)(out, a, b, c, count);
}

void Xor3(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
          const int64_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(Xor3Int64)(out, a, b, c, count);
}

void Or3(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
         const int32_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(Or3Int32)(out, a, b, c, count);
}

void Or3(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
         const int64_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(Or3Int64)(out, a, b, c, count);
}

void OrAnd(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
           const int32_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(OrAndInt32)(out, a, b, c, count);
}

void OrAnd(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT a, const int64_t* HWY_RESTRICT b,
           const int64_t* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(OrAndInt64)(out, a, b, c, count);
}

void ReverseBits(uint32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseBitsUint32)(out, a, count);
}

void ReverseBits(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReverseBitsUint64)(out, a, count);
}

void HighestSetBitIndex(int32_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(HighestSetBitIndexUint32)(out, a, count);
}

void HighestSetBitIndex(int32_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(HighestSetBitIndexUint64)(out, a, count);
}

// =============================================================================
// P2: Memory Operations
// =============================================================================

void LoadDup128(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadDup128Float32)(out, src, count);
}

void LoadDup128(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(LoadDup128Float64)(out, src, count);
}

void GatherOffset(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                  const int32_t* HWY_RESTRICT offsets, size_t count) {
    HWY_DYNAMIC_DISPATCH(GatherOffsetFloat32)(out, base, offsets, count);
}

void GatherOffset(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                  const int32_t* HWY_RESTRICT offsets, size_t count) {
    HWY_DYNAMIC_DISPATCH(GatherOffsetInt32)(out, base, offsets, count);
}

void ScatterOffset(float* HWY_RESTRICT base, const float* HWY_RESTRICT src,
                   const int32_t* HWY_RESTRICT offsets, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScatterOffsetFloat32)(base, src, offsets, count);
}

void ScatterOffset(int32_t* HWY_RESTRICT base, const int32_t* HWY_RESTRICT src,
                   const int32_t* HWY_RESTRICT offsets, size_t count) {
    HWY_DYNAMIC_DISPATCH(ScatterOffsetInt32)(base, src, offsets, count);
}

void MaskedGatherIndex(float* HWY_RESTRICT out, const float* HWY_RESTRICT base,
                       const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                       float fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGatherIndexFloat32)(out, base, indices, mask, fallback, count);
}

void MaskedGatherIndex(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT base,
                       const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                       int32_t fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGatherIndexInt32)(out, base, indices, mask, fallback, count);
}

void MaskedScatterIndex(float* HWY_RESTRICT base, const float* HWY_RESTRICT src,
                        const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                        size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedScatterIndexFloat32)(base, src, indices, mask, count);
}

void MaskedScatterIndex(int32_t* HWY_RESTRICT base, const int32_t* HWY_RESTRICT src,
                        const int32_t* HWY_RESTRICT indices, const uint8_t* HWY_RESTRICT mask,
                        size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedScatterIndexInt32)(base, src, indices, mask, count);
}

void SafeFillN(float* HWY_RESTRICT dst, float value, size_t count) {
    HWY_DYNAMIC_DISPATCH(SafeFillNFloat32)(dst, value, count);
}

void SafeFillN(int32_t* HWY_RESTRICT dst, int32_t value, size_t count) {
    HWY_DYNAMIC_DISPATCH(SafeFillNInt32)(dst, value, count);
}

void SafeCopyN(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(SafeCopyNFloat32)(dst, src, count);
}

void SafeCopyN(int32_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(SafeCopyNInt32)(dst, src, count);
}

// =============================================================================
// P2: Special Operations
// =============================================================================

void MulByPow2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT exp, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulByPow2Float32)(out, a, exp, count);
}

void MulByPow2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
               const int32_t* HWY_RESTRICT exp, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulByPow2Float64)(out, a, exp, count);
}

void GetExponent(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(GetExponentFloat32)(out, a, count);
}

void GetExponent(int32_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(GetExponentFloat64)(out, a, count);
}

void SignBit(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SignBitFloat32)(out, a, count);
}

void SignBit(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT a, size_t count) {
    HWY_DYNAMIC_DISPATCH(SignBitFloat64)(out, a, count);
}

void NaN(float* HWY_RESTRICT out, size_t count) {
    HWY_DYNAMIC_DISPATCH(NaNFloat32)(out, count);
}

void NaN(double* HWY_RESTRICT out, size_t count) {
    HWY_DYNAMIC_DISPATCH(NaNFloat64)(out, count);
}

void Inf(float* HWY_RESTRICT out, size_t count) {
    HWY_DYNAMIC_DISPATCH(InfFloat32)(out, count);
}

void Inf(double* HWY_RESTRICT out, size_t count) {
    HWY_DYNAMIC_DISPATCH(InfFloat64)(out, count);
}

// =============================================================================
// P3: Complex Number Operations
// =============================================================================

void ComplexConj(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(ComplexConjFloat32)(out, in, count);
}

void ComplexConj(double* HWY_RESTRICT out, const double* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(ComplexConjFloat64)(out, in, count);
}

void MulComplex(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
                size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexFloat32)(out, a, b, count);
}

void MulComplex(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                const double* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexFloat64)(out, a, b, count);
}

void MulComplexAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                   const float* HWY_RESTRICT b, const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexAddFloat32)(out, a, b, c, count);
}

// =============================================================================
// P3: Saturation Operations
// =============================================================================

void SaturatedAdd(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedAddInt8)(out, a, b, count);
}

void SaturatedSub(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedSubInt8)(out, a, b, count);
}

void SaturatedAdd(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                  const uint16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedAddUint16)(out, a, b, count);
}

void SaturatedSub(uint16_t* HWY_RESTRICT out, const uint16_t* HWY_RESTRICT a,
                  const uint16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SaturatedSubUint16)(out, a, b, count);
}

// =============================================================================
// P3: Block Operations
// =============================================================================

void SlideUpBlocks(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t blocks,
                   size_t count) {
    HWY_DYNAMIC_DISPATCH(SlideUpBlocksFloat32)(out, in, blocks, count);
}

void SlideDownBlocks(float* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t blocks,
                     size_t count) {
    HWY_DYNAMIC_DISPATCH(SlideDownBlocksFloat32)(out, in, blocks, count);
}

void CombineShiftRightLanes(float* HWY_RESTRICT out, const float* HWY_RESTRICT hi,
                            const float* HWY_RESTRICT lo, size_t shift, size_t count) {
    HWY_DYNAMIC_DISPATCH(CombineShiftRightLanesFloat32)(out, hi, lo, shift, count);
}

// =============================================================================
// P3: Additional Masked Operations
// =============================================================================

void MaskedSqrt(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, float fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSqrtFloat32)(out, src, mask, fallback, count);
}

void MaskedSqrt(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                const uint8_t* HWY_RESTRICT mask, double fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSqrtFloat64)(out, src, mask, fallback, count);
}

void ZeroIfNegative(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ZeroIfNegativeFloat32)(out, src, count);
}

void ZeroIfNegative(double* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ZeroIfNegativeFloat64)(out, src, count);
}

// =============================================================================
// P2: Additional Type Conversion Operations
// =============================================================================

void ReorderDemote2To(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT hi,
                      const int32_t* HWY_RESTRICT lo, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReorderDemote2ToInt32Int16)(out, hi, lo, count);
}

void ReorderDemote2To(uint16_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT hi,
                      const uint32_t* HWY_RESTRICT lo, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReorderDemote2ToUint32Uint16)(out, hi, lo, count);
}

void OrderedTruncate2To(int16_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT hi,
                        const int32_t* HWY_RESTRICT lo, size_t count) {
    HWY_DYNAMIC_DISPATCH(OrderedTruncate2ToInt32Int16)(out, hi, lo, count);
}

void OrderedTruncate2To(uint16_t* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT hi,
                        const uint32_t* HWY_RESTRICT lo, size_t count) {
    HWY_DYNAMIC_DISPATCH(OrderedTruncate2ToUint32Uint16)(out, hi, lo, count);
}

void ConvertInRangeTo(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConvertInRangeToFloat32Int32)(out, src, count);
}

void ConvertInRangeTo(int64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ConvertInRangeToFloat64Int64)(out, src, count);
}

void ResizeBitCast(uint32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ResizeBitCastFloat32Uint32)(out, src, count);
}

void ResizeBitCast(float* HWY_RESTRICT out, const uint32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ResizeBitCastUint32Float32)(out, src, count);
}

void ResizeBitCast(uint64_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ResizeBitCastFloat64Uint64)(out, src, count);
}

void ResizeBitCast(double* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(ResizeBitCastUint64Float64)(out, src, count);
}

// =============================================================================
// P2: Additional Special Operations
// =============================================================================

void MulByFloorPow2(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                    const float* HWY_RESTRICT pow2, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulByFloorPow2Float32)(out, a, pow2, count);
}

void MulByFloorPow2(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                    const double* HWY_RESTRICT pow2, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulByFloorPow2Float64)(out, a, pow2, count);
}

void GetBiasedExponent(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(GetBiasedExponentFloat32)(out, src, count);
}

void GetBiasedExponent(int32_t* HWY_RESTRICT out, const double* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(GetBiasedExponentFloat64)(out, src, count);
}

void MulFixedPoint15(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                     const int16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulFixedPoint15Int16)(out, a, b, count);
}

void MulRound(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
              const int16_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulRoundInt16)(out, a, b, count);
}

void MulRound(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulRoundInt32)(out, a, b, count);
}

void RoundingShiftRight(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT src, int shift,
                        size_t count) {
    HWY_DYNAMIC_DISPATCH(RoundingShiftRightInt16)(out, src, shift, count);
}

void RoundingShiftRight(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, int shift,
                        size_t count) {
    HWY_DYNAMIC_DISPATCH(RoundingShiftRightInt32)(out, src, shift, count);
}

// =============================================================================
// P3: Additional Complex Number Operations
// =============================================================================

void MulComplexConj(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                    const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexConjFloat32)(out, a, b, count);
}

void MulComplexConj(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                    const double* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexConjFloat64)(out, a, b, count);
}

void MulComplexConjAdd(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                       const float* HWY_RESTRICT b, const float* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexConjAddFloat32)(out, a, b, c, count);
}

void MulComplexConjAdd(double* HWY_RESTRICT out, const double* HWY_RESTRICT a,
                       const double* HWY_RESTRICT b, const double* HWY_RESTRICT c, size_t count) {
    HWY_DYNAMIC_DISPATCH(MulComplexConjAddFloat64)(out, a, b, c, count);
}

// =============================================================================
// P3: Per-Lane Block Shuffle
// =============================================================================

void Per4LaneBlockShuffle(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, uint8_t pattern,
                          size_t count) {
    HWY_DYNAMIC_DISPATCH(Per4LaneBlockShuffleFloat32)(out, src, pattern, count);
}

void Per4LaneBlockShuffle(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                          uint8_t pattern, size_t count) {
    HWY_DYNAMIC_DISPATCH(Per4LaneBlockShuffleInt32)(out, src, pattern, count);
}

// =============================================================================
// P3: Additional Masked Operations
// =============================================================================

void MaskedReciprocal(float* HWY_RESTRICT out, const float* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, float fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedReciprocalFloat32)(out, src, mask, fallback, count);
}

void MaskedReciprocal(double* HWY_RESTRICT out, const double* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, double fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedReciprocalFloat64)(out, src, mask, fallback, count);
}

void MaskedShiftLeft(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                     const uint8_t* HWY_RESTRICT mask, int shift, int32_t fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedShiftLeftInt32)(out, src, mask, shift, fallback, count);
}

void MaskedShiftLeft(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                     const uint8_t* HWY_RESTRICT mask, int shift, int64_t fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedShiftLeftInt64)(out, src, mask, shift, fallback, count);
}

void MaskedShiftRight(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, int shift, int32_t fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedShiftRightInt32)(out, src, mask, shift, fallback, count);
}

void MaskedShiftRight(int64_t* HWY_RESTRICT out, const int64_t* HWY_RESTRICT src,
                      const uint8_t* HWY_RESTRICT mask, int shift, int64_t fallback, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedShiftRightInt64)(out, src, mask, shift, fallback, count);
}

void MaskedSatAdd(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int8_t fallback,
                  size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSatAddInt8)(out, a, b, mask, fallback, count);
}

void MaskedSatAdd(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int16_t fallback,
                  size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSatAddInt16)(out, a, b, mask, fallback, count);
}

void MaskedSatSub(int8_t* HWY_RESTRICT out, const int8_t* HWY_RESTRICT a,
                  const int8_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int8_t fallback,
                  size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSatSubInt8)(out, a, b, mask, fallback, count);
}

void MaskedSatSub(int16_t* HWY_RESTRICT out, const int16_t* HWY_RESTRICT a,
                  const int16_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, int16_t fallback,
                  size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSatSubInt16)(out, a, b, mask, fallback, count);
}

// =============================================================================
// P3: Masked Comparison Operations
// =============================================================================

void MaskedEq(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedEqFloat32)(out, a, b, mask, count);
}

void MaskedEq(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedEqInt32)(out, a, b, mask, count);
}

void MaskedNe(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedNeFloat32)(out, a, b, mask, count);
}

void MaskedNe(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedNeInt32)(out, a, b, mask, count);
}

void MaskedLt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLtFloat32)(out, a, b, mask, count);
}

void MaskedLt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLtInt32)(out, a, b, mask, count);
}

void MaskedLe(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLeFloat32)(out, a, b, mask, count);
}

void MaskedLe(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedLeInt32)(out, a, b, mask, count);
}

void MaskedGt(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGtFloat32)(out, a, b, mask, count);
}

void MaskedGt(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGtInt32)(out, a, b, mask, count);
}

void MaskedGe(uint8_t* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
              const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGeFloat32)(out, a, b, mask, count);
}

void MaskedGe(uint8_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT a,
              const int32_t* HWY_RESTRICT b, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGeInt32)(out, a, b, mask, count);
}

// =============================================================================
// P3.2: Cryptographic Operations
// =============================================================================

void AESRound(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
              const uint8_t* HWY_RESTRICT round_key, size_t count) {
    HWY_DYNAMIC_DISPATCH(AESRoundImpl)(out, state, round_key, count);
}

void AESLastRound(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                  const uint8_t* HWY_RESTRICT round_key, size_t count) {
    HWY_DYNAMIC_DISPATCH(AESLastRoundImpl)(out, state, round_key, count);
}

void AESRoundInv(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                 const uint8_t* HWY_RESTRICT round_key, size_t count) {
    HWY_DYNAMIC_DISPATCH(AESRoundInvImpl)(out, state, round_key, count);
}

void AESLastRoundInv(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state,
                     const uint8_t* HWY_RESTRICT round_key, size_t count) {
    HWY_DYNAMIC_DISPATCH(AESLastRoundInvImpl)(out, state, round_key, count);
}

void AESInvMixColumns(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT state, size_t count) {
    HWY_DYNAMIC_DISPATCH(AESInvMixColumnsImpl)(out, state, count);
}

void AESKeyGenAssist(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT key, uint8_t rcon,
                     size_t count) {
    HWY_DYNAMIC_DISPATCH(AESKeyGenAssistImpl)(out, key, rcon, count);
}

void CLMulLower(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                const uint64_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(CLMulLowerImpl)(out, a, b, count);
}

void CLMulUpper(uint64_t* HWY_RESTRICT out, const uint64_t* HWY_RESTRICT a,
                const uint64_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(CLMulUpperImpl)(out, a, b, count);
}

// =============================================================================
// P3.3: Random Number Generation
// =============================================================================

void RandomStateInit(uint64_t* state, uint64_t seed) {
    HWY_DYNAMIC_DISPATCH(RandomStateInitImpl)(state, seed);
}

void Random32(uint32_t* HWY_RESTRICT out, uint64_t* state, size_t count) {
    HWY_DYNAMIC_DISPATCH(Random32Impl)(out, state, count);
}

void Random64(uint64_t* HWY_RESTRICT out, uint64_t* state, size_t count) {
    HWY_DYNAMIC_DISPATCH(Random64Impl)(out, state, count);
}

void RandomFloat(float* HWY_RESTRICT out, uint64_t* state, size_t count) {
    HWY_DYNAMIC_DISPATCH(RandomFloatImpl)(out, state, count);
}

// =============================================================================
// P3.4: Bit Packing
// =============================================================================

void PackBits(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(PackBitsImpl)(out, src, count);
}

void UnpackBits(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(UnpackBitsImpl)(out, src, count);
}

// =============================================================================
// P3.6: Algorithm Operations
// =============================================================================

size_t FindIfGreaterThan(const float* HWY_RESTRICT arr, float threshold, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindIfGreaterThanFloat32)(arr, threshold, count);
}

size_t FindIfGreaterThan(const int32_t* HWY_RESTRICT arr, int32_t threshold, size_t count) {
    return HWY_DYNAMIC_DISPATCH(FindIfGreaterThanInt32)(arr, threshold, count);
}

void Generate(float* HWY_RESTRICT out, float start, float step, size_t count) {
    HWY_DYNAMIC_DISPATCH(GenerateFloat32)(out, start, step, count);
}

void Generate(int32_t* HWY_RESTRICT out, int32_t start, int32_t step, size_t count) {
    HWY_DYNAMIC_DISPATCH(GenerateInt32)(out, start, step, count);
}

void Replace(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, float old_val, float new_val,
             size_t count) {
    HWY_DYNAMIC_DISPATCH(ReplaceFloat32)(out, src, old_val, new_val, count);
}

void Replace(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src, int32_t old_val,
             int32_t new_val, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReplaceInt32)(out, src, old_val, new_val, count);
}

void ReplaceIfGreaterThan(float* HWY_RESTRICT out, const float* HWY_RESTRICT src, float threshold,
                          float new_val, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReplaceIfGreaterThanFloat32)(out, src, threshold, new_val, count);
}

void ReplaceIfGreaterThan(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT src,
                          int32_t threshold, int32_t new_val, size_t count) {
    HWY_DYNAMIC_DISPATCH(ReplaceIfGreaterThanInt32)(out, src, threshold, new_val, count);
}

// =============================================================================
// P3.5: Image Processing Operations
// =============================================================================

ImageF ImageCreate(size_t width, size_t height) {
    ImageF img;
    img.width = width;
    img.height = height;
    // Align stride to 64 bytes for optimal SIMD performance
    img.stride = (width + 15) & ~15u;  // Round up to multiple of 16
    // Use standard aligned allocation (64-byte alignment for AVX-512)
    size_t total_bytes = img.stride * height * sizeof(float);
    img.data = static_cast<float*>(std::aligned_alloc(64, total_bytes));
    return img;
}

void ImageFree(ImageF* img) {
    if (img && img->data) {
        std::free(img->data);
        img->data = nullptr;
        img->width = 0;
        img->height = 0;
        img->stride = 0;
    }
}

void ImageFill(ImageF* img, float value) {
    HWY_DYNAMIC_DISPATCH(ImageFillImpl)(img->data, img->width, img->height, img->stride, value);
}

void ImageCopy(const ImageF* src, ImageF* dst) {
    HWY_DYNAMIC_DISPATCH(ImageCopyImpl)
    (src->data, src->stride, dst->data, dst->stride, src->width, src->height);
}

void ImageAdd(const ImageF* a, const ImageF* b, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(ImageAddImpl)
    (a->data, a->stride, b->data, b->stride, out->data, out->stride, a->width, a->height);
}

void ImageSub(const ImageF* a, const ImageF* b, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(ImageSubImpl)
    (a->data, a->stride, b->data, b->stride, out->data, out->stride, a->width, a->height);
}

void ImageMul(const ImageF* a, const ImageF* b, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(ImageMulImpl)
    (a->data, a->stride, b->data, b->stride, out->data, out->stride, a->width, a->height);
}

void ImageScale(const ImageF* src, float scale, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(ImageScaleImpl)
    (src->data, src->stride, scale, out->data, out->stride, src->width, src->height);
}

void ImageClamp(const ImageF* src, float min_val, float max_val, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(ImageClampImpl)
    (src->data, src->stride, min_val, max_val, out->data, out->stride, src->width, src->height);
}

void Convolve3x3(const ImageF* src, const float* kernel, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(Convolve3x3Impl)
    (src->data, src->stride, kernel, out->data, out->stride, src->width, src->height);
}

void BoxBlur3x3(const ImageF* src, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(BoxBlur3x3Impl)
    (src->data, src->stride, out->data, out->stride, src->width, src->height);
}

void GaussianBlur3x3(const ImageF* src, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(GaussianBlur3x3Impl)
    (src->data, src->stride, out->data, out->stride, src->width, src->height);
}

void SobelEdge(const ImageF* src, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(SobelEdgeImpl)
    (src->data, src->stride, out->data, out->stride, src->width, src->height);
}

void Sharpen(const ImageF* src, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(SharpenImpl)
    (src->data, src->stride, out->data, out->stride, src->width, src->height);
}

void Threshold(const ImageF* src, float threshold, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(ThresholdImpl)
    (src->data, src->stride, threshold, out->data, out->stride, src->width, src->height);
}

void Grayscale(const ImageF* r, const ImageF* g, const ImageF* b, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(GrayscaleImpl)
    (r->data, r->stride, g->data, g->stride, b->data, b->stride, out->data, out->stride, r->width,
     r->height);
}

void Downsample2x(const ImageF* src, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(Downsample2xImpl)
    (src->data, src->stride, out->data, out->stride, out->width, out->height);
}

void Upsample2x(const ImageF* src, ImageF* out) {
    HWY_DYNAMIC_DISPATCH(Upsample2xImpl)
    (src->data, src->stride, src->width, src->height, out->data, out->stride);
}

// =============================================================================
// Gap Operations: Additional SIMD Operations
// =============================================================================

void MaskedMulAdd(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                  const float* HWY_RESTRICT mul, const float* HWY_RESTRICT x,
                  const float* HWY_RESTRICT add, const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMulAddFloat32)(out, mask, mul, x, add, no, count);
}

void MaskedNegMulAdd(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                     const float* HWY_RESTRICT mul, const float* HWY_RESTRICT x,
                     const float* HWY_RESTRICT add, const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedNegMulAddFloat32)(out, mask, mul, x, add, no, count);
}

void InterleaveWholeLower(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveWholeLowerFloat32)(out, a, b, count);
}

void InterleaveWholeUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(InterleaveWholeUpperFloat32)(out, a, b, count);
}

void OddEven(float* HWY_RESTRICT out, const float* HWY_RESTRICT odd, const float* HWY_RESTRICT even,
             size_t count) {
    HWY_DYNAMIC_DISPATCH(OddEvenFloat32)(out, odd, even, count);
}

void OddEven(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT odd,
             const int32_t* HWY_RESTRICT even, size_t count) {
    HWY_DYNAMIC_DISPATCH(OddEvenInt32)(out, odd, even, count);
}

void IfNegativeThenElseZero(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                            const float* HWY_RESTRICT yes, size_t count) {
    HWY_DYNAMIC_DISPATCH(IfNegativeThenElseZeroFloat32)(out, v, yes, count);
}

void IfNegativeThenZeroElse(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                            const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(IfNegativeThenZeroElseFloat32)(out, v, no, count);
}

void BitwiseIfThenElse(int32_t* HWY_RESTRICT out, const int32_t* HWY_RESTRICT mask,
                       const int32_t* HWY_RESTRICT yes, const int32_t* HWY_RESTRICT no,
                       size_t count) {
    HWY_DYNAMIC_DISPATCH(BitwiseIfThenElseInt32)(out, mask, yes, no, count);
}

void MaskFalse(uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskFalseImpl)(mask, count);
}

void SetMask(uint8_t* HWY_RESTRICT mask, bool value, size_t count) {
    HWY_DYNAMIC_DISPATCH(SetMaskImpl)(mask, value, count);
}

void CeilInt(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(CeilIntFloat32)(out, in, count);
}

void FloorInt(int32_t* HWY_RESTRICT out, const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(FloorIntFloat32)(out, in, count);
}

void TruncateStore(int16_t* HWY_RESTRICT dst, const int32_t* HWY_RESTRICT src, size_t count) {
    HWY_DYNAMIC_DISPATCH(TruncateStoreInt32ToInt16)(dst, src, count);
}

void MaskedModOr(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                 const int32_t* HWY_RESTRICT a, const int32_t* HWY_RESTRICT b,
                 const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedModOrInt32)(out, mask, a, b, no, count);
}

// =============================================================================
// Final Gap Operations
// =============================================================================

void Per2LaneBlockShuffle(float* HWY_RESTRICT out, const float* HWY_RESTRICT a,
                          const float* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(Per2LaneBlockShuffleFloat32)(out, a, b, count);
}

void MaskedIsNaN(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                 const float* HWY_RESTRICT in, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedIsNaNFloat32)(out, mask, in, count);
}

void IfNegativeThenNegOrUndefIfZero(float* HWY_RESTRICT out, const float* HWY_RESTRICT v,
                                    const float* HWY_RESTRICT x, size_t count) {
    HWY_DYNAMIC_DISPATCH(IfNegativeThenNegOrUndefIfZeroFloat32)(out, v, x, count);
}

void MaskedSetOr(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, float value,
                 const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedSetOrFloat32)(out, mask, value, no, count);
}

void MaskedMulFixedPoint15(int16_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                           const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
                           const int16_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedMulFixedPoint15Int16)(out, mask, a, b, no, count);
}

void MaskedWidenMulPairwiseAdd(int32_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                               const int16_t* HWY_RESTRICT a, const int16_t* HWY_RESTRICT b,
                               const int32_t* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedWidenMulPairwiseAddInt16ToInt32)(out, mask, a, b, no, count);
}

void MaskedAbsOr(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                 const float* HWY_RESTRICT a, const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedAbsOrFloat32)(out, mask, a, no, count);
}

void InsertIntoUpper(float* HWY_RESTRICT out, const float* HWY_RESTRICT vec, float scalar,
                     size_t count) {
    HWY_DYNAMIC_DISPATCH(InsertIntoUpperFloat32)(out, vec, scalar, count);
}

void MaskedGatherIndexOr(float* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask,
                         const float* HWY_RESTRICT base, const int32_t* HWY_RESTRICT indices,
                         const float* HWY_RESTRICT no, size_t count) {
    HWY_DYNAMIC_DISPATCH(MaskedGatherIndexOrFloat32)(out, mask, base, indices, no, count);
}

void SumsOf8AbsDiff(uint64_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT a,
                    const uint8_t* HWY_RESTRICT b, size_t count) {
    HWY_DYNAMIC_DISPATCH(SumsOf8AbsDiffUint8)(out, a, b, count);
}

void CombineMasks(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT lo,
                  const uint8_t* HWY_RESTRICT hi, size_t half_count) {
    HWY_DYNAMIC_DISPATCH(CombineMasksImpl)(out, lo, hi, half_count);
}

void LowerHalfOfMask(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(LowerHalfOfMaskImpl)(out, mask, count);
}

void UpperHalfOfMask(uint8_t* HWY_RESTRICT out, const uint8_t* HWY_RESTRICT mask, size_t count) {
    HWY_DYNAMIC_DISPATCH(UpperHalfOfMaskImpl)(out, mask, count);
}

void PromoteMaskTo(uint16_t* HWY_RESTRICT wide, const uint8_t* HWY_RESTRICT narrow, size_t count) {
    HWY_DYNAMIC_DISPATCH(PromoteMaskToUint8ToUint16)(wide, narrow, count);
}

void DemoteMaskTo(uint8_t* HWY_RESTRICT narrow, const uint16_t* HWY_RESTRICT wide, size_t count) {
    HWY_DYNAMIC_DISPATCH(DemoteMaskToUint16ToUint8)(narrow, wide, count);
}

void ZeroExtendResizeBitCast(uint32_t* HWY_RESTRICT dst, const uint8_t* HWY_RESTRICT src,
                             size_t count) {
    HWY_DYNAMIC_DISPATCH(ZeroExtendResizeBitCastUint8ToUint32)(dst, src, count);
}

// Note: F64ToF16 and F64ToBF16 already defined above

void MatVecMul(float* HWY_RESTRICT out, const float* HWY_RESTRICT mat,
               const float* HWY_RESTRICT vec, size_t rows, size_t cols) {
    HWY_DYNAMIC_DISPATCH(MatVecMulFloat32)(out, mat, vec, rows, cols);
}

void MatMul(float* HWY_RESTRICT out, const float* HWY_RESTRICT a, const float* HWY_RESTRICT b,
            size_t M, size_t K, size_t N) {
    HWY_DYNAMIC_DISPATCH(MatMulFloat32)(out, a, b, M, K, N);
}

}  // namespace simd
}  // namespace bud
