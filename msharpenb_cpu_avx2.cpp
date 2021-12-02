/*
**						MSharpena Filter AVX2 and CUDA ver 0.0.2
**
**	The MIT License
**
**	Copyright (c) 2012-2014 miya <@sundola8x>
**
**	Permission is hereby granted, free of charge, to any person obtaining a copy of
**	this software and associated documentation files (the "Software"), to deal in
**	the Software without restriction, including without limitation the rights to
**	use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
**	of the Software, and to permit persons to whom the Software is furnished to do
**	so, subject to the following conditions:
**
**	The above copyright notice and this permission notice shall be included in all
**	copies or substantial portions of the Software.
**
**	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
**	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
**	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
**	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
**	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
**	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
**	SOFTWARE.
**
**	Thanks to the original plugin author: https://github.com/tp7/msharpen
*/
#include "stdafx.h"
#include "msharpenb.h"
#include "msharpenb_cpu.h"

#define MSHARPEN_TRUE		0xffff
#define MSHARPEN_FALSE		0
#define BLUR_CENTER_MUL		22444
#define BLUR_ANOTHER_MUL	21546

#define MSHAPERN_COPY_WORK_INSRW(idx) \
vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], idx;\
vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], idx;\
vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], idx; 

typedef struct tagMSHARPEN_AVX2_CORE_PARAM
{
	// 共有メモリの設定 
	int thread_step;				// 0 - 4
	// - ブレーンコピー用
	short	*work;					// 4 - 8
	// - ブラー作業用
	short	*wblurx;				// 8 - 12
	// - ブラー作業用2
	short	*wblury;				// 12 - 16
	// - マスク作業用	
	unsigned short   *wmask;					// 16 - 20
	//2014.09.01 ↓SIMD向けに実相
	// - ブラー作業用横ピッチ
	int		wpitch;					// 20 - 24
	// - プレーン選択用
	int		zpitch;					// 24 - 28

	int		y__plane;				// 28 - 32
	int		cb_plane;				// 32 - 36
	int		cr_plane;				// 36 - 40

	// -- filler
	int		hoge[6];				// 40 - 64

	// -- mm
	short	threshold_mm[16];
	short	true_bit_pattern_mm[16];
	short	blur_center_mul_mm[16];
	short	blur_another_mul_mm[16];
	short	strength_mm[16];
	short	invstrength_mm[16];
	short	detect_max_mm[16];
	short	detect_min_mm[16];

	//2014.09.01 ↑SIMD向けに実相

} MSHARPEN_AVX2_CORE_PARAM, *LPMSHARPEN_AVX2_CORE_PARAM;

#pragma region avx2 filter

//---------------------------------------------------------------------
//		インライン関数
//---------------------------------------------------------------------
// 閾値チェック
inline BOOL is_threshold_avx2_over(_In_ const short *yca, _In_ const short *ycb, const LPMSHARPEN_AVX2_CORE_PARAM core_param) {
	if (abs(yca[core_param->y__plane] - ycb[core_param->y__plane]) >= param.threshold) return TRUE;
	if (abs(yca[core_param->cb_plane] - ycb[core_param->cb_plane]) >= param.threshold) return TRUE;
	if (abs(yca[core_param->cr_plane] - ycb[core_param->cr_plane]) >= param.threshold) return TRUE;
	return FALSE;
}

//---------------------------------------------------------------------
//		フィルタ処理関数 simd予定
//---------------------------------------------------------------------
void msharpen_avx2_core(
	_In_ int thread_id,
	_In_ int thread_num,
	_In_ void *param1,
	_In_ void *param2)
{
	//========================================
	// チェック
	//========================================
	FILTER_PROC_INFO *fpip = (FILTER_PROC_INFO *)param2;
	assert(fpip != NULL);
	assert(fpip->ycp_edit != NULL);
	assert(fpip->ycp_temp != NULL);

	// ========================================
	// マルチスレッドでの処理分割
	// ========================================
	int y_start = (fpip->h * thread_id) / thread_num;
	int y_end = (fpip->h * (thread_id + 1)) / thread_num;

	//========================================
	// 初期化
	//========================================
	int max_w = fpip->max_w;						// 画像領域のサイズ 横幅
	MSHARPEN_AVX2_CORE_PARAM* core_param = (MSHARPEN_AVX2_CORE_PARAM*)param1;
	assert(core_param != NULL);
	assert(core_param->work != NULL);
	assert(core_param->wblurx != NULL);
	assert(core_param->wblury != NULL);
	assert(core_param->wmask != NULL);
//		MSHARPEN_AVX2_CORE_PARAM core;
//		memcpy(&core, core_param, sizeof(MSHARPEN_AVX2_CORE_PARAM));
	int		w = fpip->w;
	int		wpitch = core_param->wpitch;
	int		zpitch = core_param->zpitch;
	int		y__plane = core_param->y__plane;
	int		cb_plane = core_param->cb_plane;
	int		cr_plane = core_param->cr_plane;
	// プレーンコピー用目メモリ
	short	*work   = core_param->work + core_param->thread_step * thread_id * wpitch;
	// ブラー用作業メモリ
	short	*wblurx = core_param->wblurx + core_param->thread_step * thread_id * wpitch;
	short	*wblury = core_param->wblury + core_param->thread_step * thread_id * wpitch;
	// マーキング用作業メモリの確保
	unsigned short	*wmask = core_param->wmask + core_param->thread_step * thread_id * wpitch;

	// -- mm --
	_declspec(align(32)) short	blur_center_mul_mm[16];
	_declspec(align(32)) short	blur_another_mul_mm[16];
	_declspec(align(32)) short	threshold_mm[16];
	_declspec(align(32)) short	true_bit_pattern_mm[16];
	_declspec(align(32)) short	strength_mm[16];
	_declspec(align(32)) short	invstrength_mm[16];
	_declspec(align(32)) short	detect_max_mm[16];
	_declspec(align(32)) short	detect_min_mm[16];

	memcpy(blur_center_mul_mm, core_param->blur_center_mul_mm, sizeof(short) * 16);
	memcpy(blur_another_mul_mm, core_param->blur_another_mul_mm, sizeof(short) * 16);
	memcpy(threshold_mm, core_param->threshold_mm, sizeof(short) * 16);
	memcpy(true_bit_pattern_mm, core_param->true_bit_pattern_mm, sizeof(short) * 16);
	memcpy(strength_mm, core_param->strength_mm, sizeof(short) * 16);
	memcpy(invstrength_mm, core_param->invstrength_mm, sizeof(short) * 16);
	memcpy(detect_max_mm, core_param->detect_max_mm, sizeof(short) * 16);
	memcpy(detect_min_mm, core_param->detect_min_mm, sizeof(short) * 16);


	//========================================
	// プレーンコピー　要SIMD化、断念
	//========================================
	DBG("[%-16s] [thread(%d)] ========== copy_work ==========\n", __FUNCTION__, thread_id);
	int simdcount = 0;

	int y_start1 = y_start - BLURX_MARGIN_Y;	// 縦ブラーマージン 
	int y_end1 = y_end + BLURX_MARGIN_Y;	// 縦ブラーマージン 
	if (y_start1 <  0) y_start1 = 0;
	if (y_end1 >= fpip->h) y_end1 = fpip->h;
	for (int y = y_start1; y < y_end1; y++) {
		// 処理対象のポインタの計算
		PIXEL_YC* ycp = fpip->ycp_edit + y * max_w;		// 画像データ
		short* workp = work + (y - y_start + BLURX_MARGIN_Y) * wpitch;

		int x = 0;
		// -- simd域 --
		/*
		simdcount = fpip->w / 32;	//8*4*2=64バイト＝512ビット short型なので2分の1
		for (int i = 0; i < simdcount; i++) {
			_asm {
				push		esi;
				push		edi;
				push		edx;
				push		ecx;
				push		eax;

				// -- load pointer --
				mov			esi, DWORD PTR[ycp];			// load hoge001(esi) ycp pointer
				mov			edi, DWORD PTR[workp];			// load hoge002(edi) workp pointer

				// -- load --
				mov			eax, DWORD PTR[zpitch];			// load hoge003(eax) zpitch

				// -- y__plane load sec --
				xor			edx, edx;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 0;		// Y00,---|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 0;		// Y08,---|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 0;		// Y16,---|---,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 0;	// Y24,---|---,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 1;		// Y00,Y01|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 1;		// Y08,Y09|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 1;		// Y16,Y17|---,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 1;	// Y24,Y25|---,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 2;		// Y00,Y01|Y02,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 2;		// Y08,Y09|Y10,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 2;		// Y16,Y17|Y18,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 2;	// Y24,Y25|Y26,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 3;		// Y00,Y01|Y02,Y03|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 3;		// Y08,Y09|Y10,Y11|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 3;		// Y16,Y17|Y18,Y19|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 3;	// Y24,Y25|Y26,Y27|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 4;		// Y00,Y01|Y02,Y03|Y04,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 4;		// Y08,Y09|Y10,Y11|Y12,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 4;		// Y16,Y17|Y18,Y19|Y20,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 4;	// Y24,Y25|Y26,Y27|Y28,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 5;		// Y00,Y01|Y02,Y03|Y04,Y05|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 5;		// Y08,Y09|Y10,Y11|Y12,Y13|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 5;		// Y16,Y17|Y18,Y19|Y20,Y21|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 5;	// Y24,Y25|Y26,Y27|Y28,Y29|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 6;		// Y00,Y01|Y02,Y03|Y04,Y05|Y06,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 6;		// Y08,Y09|Y10,Y11|Y12,Y13|Y14,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 6;		// Y16,Y17|Y18,Y19|Y20,Y21|Y22,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 6;	// Y24,Y25|Y26,Y27|Y28,Y29|Y30,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 7;		// Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 7;		// Y08,Y09|Y10,Y11|Y12,Y13|Y14,Y15
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 7;		// Y16,Y17|Y18,Y19|Y20,Y21|Y22,Y23
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 7;	// Y24,Y25|Y26,Y27|Y28,Y29|Y30,Y31

				// -- y__plane store sec --
				vmovdqa		XMMWORD PTR[edi], xmm0;				    		// save : Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vmovdqa		XMMWORD PTR[edi + 16], xmm1;					// save : Y08,Y09|Y10,Y11|Y12,Y13|Y14,Y15
				vmovdqa		XMMWORD PTR[edi + 32], xmm2;					// save : Y16,Y17|Y18,Y19|Y20,Y21|Y22,Y23
				vmovdqa		XMMWORD PTR[edi + 48], xmm3;					// save : Y24,Y25|Y26,Y27|Y28,Y29|Y30,Y31


				// -- cb_plane load sec --
				mov			edx, 2;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 0;		// U00,---|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 0;		// U08,---|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 0;		// U16,---|---,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 0;	// U24,---|---,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 1;		// U00,U01|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 1;		// U08,U09|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 1;		// U16,U17|---,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 1;	// U24,U25|---,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 2;		// U00,U01|U02,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 2;		// U08,U09|U10,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 2;		// U16,U17|U18,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 2;	// U24,U25|U26,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 3;		// U00,U01|U02,U03|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 3;		// U08,U09|U10,U11|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 3;		// U16,U17|U18,U19|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 3;	// U24,U25|U26,U27|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 4;		// U00,U01|U02,U03|U04,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 4;		// U08,U09|U10,U11|U12,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 4;		// U16,U17|U18,U19|U20,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 4;	// U24,U25|U26,U27|U28,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 5;		// U00,U01|U02,U03|U04,U05|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 5;		// U08,U09|U10,U11|U12,U13|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 5;		// U16,U17|U18,U19|U20,U21|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 5;	// U24,U25|U26,U27|U28,U29|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 6;		// U00,U01|U02,U03|U04,U05|U06,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 6;		// U08,U09|U10,U11|U12,U13|U14,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 6;		// U16,U17|U18,U19|U20,U21|U22,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 6;	// U24,U25|U26,U27|U28,U29|U30,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 7;		// U00,U01|U02,U03|U04,U05|U06,U07
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 7;		// U08,U09|U10,U11|U12,U13|U14,U15
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 7;		// U16,U17|U18,U19|U20,U21|U22,U23
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 7;	// U24,U25|U26,U27|U28,U29|U30,U31

				// -- cb_plane store sec --
				mov			ecx, eax;
				add			ecx, ecx;
				vmovdqa		XMMWORD PTR[edi + ecx], xmm0;					// save : U00,U01|U02,U03|U04,U05|U06,U07
				vmovdqa		XMMWORD PTR[edi + ecx + 16], xmm1;				// save : U08,U09|U10,U11|U12,U13|U14,U15
				vmovdqa		XMMWORD PTR[edi + ecx + 32], xmm2;				// save : U16,U17|U18,U19|U20,U21|U22,U23
				vmovdqa		XMMWORD PTR[edi + ecx + 48], xmm3;				// save : U24,U25|U26,U27|U28,U29|U30,U31

				// -- cr_plane load sec --
				mov			edx, 4;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 0;		// V00,---|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 0;		// V08,---|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 0;		// V16,---|---,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 0;	// V24,---|---,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 1;		// V00,V01|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 1;		// V08,V09|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 1;		// V16,V17|---,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 1;	// V24,V25|---,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 2;		// V00,V01|V02,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 2;		// V08,V09|V10,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 2;		// V16,V17|V18,---|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 2;	// V24,V25|V26,---|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 3;		// V00,V01|V02,V03|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 3;		// V08,V09|V10,V11|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 3;		// V16,V17|V18,V19|---,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 3;	// V24,V25|V26,V27|---,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 4;		// V00,V01|V02,V03|V04,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 4;		// V08,V09|V10,V11|V12,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 4;		// V16,V17|V18,V19|V20,---|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 4;	// V24,V25|V26,V27|V28,---|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 5;		// V00,V01|V02,V03|V04,V05|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 5;		// V08,V09|V10,V11|V12,V13|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 5;		// V16,V17|V18,V19|V20,V21|---,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 5;	// V24,V25|V26,V27|V28,V29|---,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 6;		// V00,V01|V02,V03|V04,V05|V06,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 6;		// V08,V09|V10,V11|V12,V13|V14,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 6;		// V16,V17|V18,V19|V20,V21|V22,---
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 6;	// V24,V25|V26,V27|V28,V29|V30,---
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 7;		// V00,V01|V02,V03|V04,V05|V06,V07
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 48], 7;		// V08,V09|V10,V11|V12,V13|V14,V15
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 96], 7;		// V16,V17|V18,V19|V20,V21|V22,V23
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 144], 7;	// V24,V25|V26,V27|V28,V29|V30,V31

				// -- cr_plane store sec --
				mov			ecx, eax;
				shl			ecx, 2;
				vmovdqa		XMMWORD PTR[edi + ecx], xmm0;					// save : V00,V01|V02,V03|V04,V05|V06,V07
				vmovdqa		XMMWORD PTR[edi + ecx + 16], xmm1;				// save : V08,V09|V10,V11|V12,V13|V14,V15
				vmovdqa		XMMWORD PTR[edi + ecx + 32], xmm2;				// save : V16,V17|V18,V19|V20,V21|V22,V23
				vmovdqa		XMMWORD PTR[edi + ecx + 48], xmm3;				// save : V24,V25|V26,V27|V28,V29|V30,V31

				pop			eax;
				pop			ecx;
				pop			edx;
				pop			edi;
				pop			esi;
			}
			x += 32;
			ycp += 32;
			workp += 32;
		}
		*/
		/*
		simdcount = fpip->w / 16;	//8*2=16バイト＝128ビット short型なので2分の1
		for (int i = 0; i < simdcount; i++) {
			_asm {
				push		esi;
				push		edi;
				push		edx;
				push		eax;

				// -- load pointer --
				mov			esi, DWORD PTR[ycp];			// load hoge001(esi) ycp pointer
				mov			edi, DWORD PTR[workp];			// load hoge002(edi) workp pointer

				// -- load --
				mov			eax, DWORD PTR[zpitch];			// load hoge003(eax) zpitch


				// -- 00 pixel load sec --
				xor			edx, edx;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 0;		// Y00,---|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 0;		// U00,---|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 0;		// V00,---|---,---|---,---|---,---
				// -- 01 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 1;		// Y00,Y01|---,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 1;		// U00,U01|---,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 1;		// V00,V01|---,---|---,---|---,---
				// -- 02 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 2;		// Y00,Y01|Y02,---|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 2;		// U00,U01|U02,---|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 2;		// V00,V01|V02,---|---,---|---,---
				// -- 03 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 3;		// Y00,Y01|Y02,Y03|---,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 3;		// U00,U01|U02,U03|---,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 3;		// V00,V01|V02,V03|---,---|---,---
				// -- 04 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 4;		// Y00,Y01|Y02,Y03|Y04,---|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 4;		// U00,U01|U02,U03|U04,---|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 4;		// V00,V01|V02,V03|V04,---|---,---
				// -- 05 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 5;		// Y00,Y01|Y02,Y03|Y04,Y05|---,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 5;		// U00,U01|U02,U03|U04,U05|---,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 5;		// V00,V01|V02,V03|V04,V05|---,---
				// -- 06 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 6;		// Y00,Y01|Y02,Y03|Y04,Y05|Y06,---
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 6;		// U00,U01|U02,U03|U04,U05|U06,---
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 6;		// V00,V01|V02,V03|V04,V05|V06,---
				// -- 07 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm0, xmm0, XMMWORD PTR[esi + edx + 0], 7;		// Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vpinsrw		xmm1, xmm1, XMMWORD PTR[esi + edx + 2], 7;		// U00,U01|U02,U03|U04,U05|U06,U07
				vpinsrw		xmm2, xmm2, XMMWORD PTR[esi + edx + 4], 7;		// V00,V01|V02,V03|V04,V05|V06,V07

				// -- 08 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 0;		// Y00,---|---,---|---,---|---,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 0;		// U00,---|---,---|---,---|---,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 0;		// V00,---|---,---|---,---|---,---
				// -- 09 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 1;		// Y00,Y01|---,---|---,---|---,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 1;		// U00,U01|---,---|---,---|---,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 1;		// V00,V01|---,---|---,---|---,---
				// -- 10 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 2;		// Y00,Y01|Y02,---|---,---|---,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 2;		// U00,U01|U02,---|---,---|---,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 2;		// V00,V01|V02,---|---,---|---,---
				// -- 11 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 3;		// Y00,Y01|Y02,Y03|---,---|---,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 3;		// U00,U01|U02,U03|---,---|---,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 3;		// V00,V01|V02,V03|---,---|---,---
				// -- 12 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 4;		// Y00,Y01|Y02,Y03|Y04,---|---,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 4;		// U00,U01|U02,U03|U04,---|---,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 4;		// V00,V01|V02,V03|V04,---|---,---
				// -- 13 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 5;		// Y00,Y01|Y02,Y03|Y04,Y05|---,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 5;		// U00,U01|U02,U03|U04,U05|---,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 5;		// V00,V01|V02,V03|V04,V05|---,---
				// -- 14 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 6;		// Y00,Y01|Y02,Y03|Y04,Y05|Y06,---
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 6;		// U00,U01|U02,U03|U04,U05|U06,---
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 6;		// V00,V01|V02,V03|V04,V05|V06,---
				// -- 15 pixel load sec --
				add			edx, 6;
				vpinsrw		xmm3, xmm3, XMMWORD PTR[esi + edx + 0], 7;		// Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vpinsrw		xmm4, xmm4, XMMWORD PTR[esi + edx + 2], 7;		// U00,U01|U02,U03|U04,U05|U06,U07
				vpinsrw		xmm5, xmm5, XMMWORD PTR[esi + edx + 4], 7;		// V00,V01|V02,V03|V04,V05|V06,V07


				// -- store sec --
				vmovdqa		XMMWORD PTR[edi], xmm0;				    		// save : Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vmovdqa		XMMWORD PTR[edi + 16], xmm3;
				vmovdqa		XMMWORD PTR[edi + eax * 2], xmm1;				// save : U00,U01|U02,U03|U04,U05|U06,U07
				vmovdqa		XMMWORD PTR[edi + eax * 2 + 16], xmm4;
				vmovdqa		XMMWORD PTR[edi + eax * 4], xmm2;				// save : V00,V01|V02,V03|V04,V05|V06,V07
				vmovdqa		XMMWORD PTR[edi + eax * 4 + 16], xmm5;

				pop			eax;
				pop			edx;
				pop			edi;
				pop			esi;
			}
			x += 16;
			ycp += 16;
			workp += 16;
		}
		*/
		// -- 非simd域 --
		for (; x < fpip->w; x++) {
			workp[y__plane] = ycp->y;
			workp[cb_plane] = ycp->cb;
			workp[cr_plane] = ycp->cr;
			ycp++;
			workp++;
		}
	}

	//========================================
	// ブラー 横　要SIMD化
	//========================================
	DBG("[%-16s] [thread(%d)] ========== blur_x ==========\n", __FUNCTION__, thread_id);
	simdcount = fpip->w / 16;	//32バイト＝256ビット short型なので2分の1
	int simdcount2 = (fpip->w - 18) / 16;	//32バイト＝256ビット short型なので2分の1
	short blur_center_mul = BLUR_CENTER_MUL;
	short blur_another_mul = BLUR_ANOTHER_MUL;
	//short blur_center_mul_mm[16];
	//short blur_another_mul_mm[16];
	_asm {
		//vpbroadcastw		ymm6, WORD PTR[blur_center_mul];
		//vmovdqu             YMMWORD PTR[blur_center_mul_mm], ymm6;
		//vpbroadcastw		ymm7, WORD PTR[blur_another_mul];
		//vmovdqu             YMMWORD PTR[blur_another_mul_mm], ymm7;

		vmovdqu				ymm6, YMMWORD PTR[     blur_center_mul_mm];
		vmovdqu				ymm7, YMMWORD PTR[     blur_another_mul_mm];
	}

	short test1[16];
	short test2[16];
	short test3[16];


	y_start1 = y_start - BLURX_MARGIN_Y;	// 縦ブラーマージン 
	y_end1 = y_end + BLURX_MARGIN_Y;	// 縦ブラーマージン 
	if (y_start1 <  0) y_start1 = 0;
	if (y_end1 >= fpip->h) y_end1 = fpip->h;
	for (int y = y_start1; y < y_end1; y++) {
		// 処理対象のポインタの計算
		short* workp = work + (y - y_start + BLURX_MARGIN_Y) * wpitch;		// 画像データ
		short* wblurxp = wblurx + (y - y_start + BLURX_MARGIN_Y) * wpitch;

		// todo:上下ブラー対象外領域
		if (y <= 1 || y >= fpip->h - 2) {
			int x = 0;
			// -- simd --
			for (int i = 0; i < simdcount; i++) {
				_asm {
					push				edi;
					push				esi;
					push				ecx;

					// -- load pointer --
					mov					edi, DWORD PTR[wblurxp];
					mov					esi, DWORD PTR[workp];

					// -- --
					mov					ecx, DWORD PTR[wpitch];

					// -- copy y__plane --
					vmovdqa             ymm0, YMMWORD PTR[esi];
					vmovdqa             YMMWORD PTR[edi], ymm0;

					// -- copy cb_plane --
					vmovdqa             ymm0, YMMWORD PTR[esi + ecx * 2];
					vmovdqa             YMMWORD PTR[edi + ecx * 2], ymm0;

					// -- copy cr_plane --
					vmovdqa             ymm0, YMMWORD PTR[esi + ecx * 4];
					vmovdqa             YMMWORD PTR[edi + ecx * 4], ymm0;

					pop					ecx;
					pop					esi;
					pop					edi;
				}
				x += 16;
				workp += 16;
				wblurxp += 16;
			}
			// -- 非simd --
			for (; x < fpip->w; x++) {
				//フィルタがかからない領域 縦の範囲チェックは除外してもよい
				wblurxp[y__plane] = workp[y__plane];
				wblurxp[cb_plane] = workp[cb_plane];
				wblurxp[cr_plane] = workp[cr_plane];
				workp++;
				wblurxp++;
			}
			continue;
		}

		int x = 0;
		//   todo:ブラー左対象外域
		for (; x < 2; x++) {
			//フィルタがかからない領域 縦の範囲チェックは除外してもよい
			wblurxp[y__plane] = workp[y__plane];
			wblurxp[cb_plane] = workp[cb_plane];
			wblurxp[cr_plane] = workp[cr_plane];
			workp++;
			wblurxp++;
		}

		//   todo:ブラー対象非SIMD領域
		for (; x < 16; x++) {
			wblurxp[y__plane] = 
				((BLUR_CENTER_MUL * (int)workp[y__plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[y__plane - 1]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[y__plane + 1]) >> 16);
			wblurxp[cb_plane] =
				((BLUR_CENTER_MUL * (int)workp[cb_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cb_plane - 1]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cb_plane + 1]) >> 16);
			wblurxp[cr_plane] =
				((BLUR_CENTER_MUL * (int)workp[cr_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cr_plane - 1]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cr_plane + 1]) >> 16);
			workp++;
			wblurxp++;
		}

		//   todo:ブラー対象SIMD領域
		for (int i = 0; i < simdcount2; i++) {
			_asm {
				push		esi;
				push		edi;
				push		ecx;
				push		eax;
				push		edx;

				// -- load pointer --
				mov			esi, DWORD PTR[workp];			// load hoge001(esi) workp pointer
				mov			edi, DWORD PTR[wblurxp];		// load hoge002(edi) wblurxp pointer

				// -- load --
				mov			ecx, DWORD PTR[zpitch];			// load hoge003(ecx) zpitch
				mov			eax, DWORD PTR[wpitch];			// load hoge003(eax) wpitch


				// -- load center value sec --
				vmovdqa		ymm0, YMMWORD PTR[esi];						// load hoge010(ymm0) center y__plane value
				vpmulhw		ymm0, ymm0, ymm6;							// hoge011(ymm0) = hoge010(ymm0) * blur_center_mul_mm(ymm6) high
				vmovdqu		ymm1, YMMWORD PTR[esi + ecx * 2];			// load hoge012(ymm1) center cb_plane[+ecx * 2] value
				vpmulhw		ymm1, ymm1, ymm6;							// hoge013(ymm1) = hoge012(ymm1) * blur_center_mul_mm(ymm6) high
				vmovdqu		ymm2, YMMWORD PTR[esi + ecx * 4];			// load hoge014(ymm2) center cr_plane[+ecx * 4] value
				vpmulhw		ymm2, ymm2, ymm6;							// hoge015(ymm2) = hoge012(ymm2) * blur_center_mul_mm(ymm6) high


				// -- load left y__plane value sec --
				lea			edx, [esi - 2];								 // hoge021 = (-2) + pointer wblurxp(esi)
				vmovdqu		ymm3, YMMWORD PTR[edx];				 		 // load hoge022(ymm3) top y__plane[hoge021] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge023(ymm3) = hoge022(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm0, ymm3, ymm0;							 // hoge024(ymm0) = hoge023(ymm3) + hoge011(ymm0)

				vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 2];			 // load hoge030(ymm3) top cb_plane[hoge021 + zpitch(ecx) * 2] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge031(ymm3) = hoge030(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm1, ymm3, ymm1;							 // hoge032(ymm1) = hoge031(ymm3) + hoge013(ymm1)

				vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 4];			 // load hoge040(ymm3) top cr_plane[hoge021 + zpitch(ecx) * 4] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge041(ymm3) = hoge040(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm2, ymm3, ymm2;							 // hoge042(ymm2) = hoge041(ymm3) + hoge015(ymm2)


				// -- load right y__plane value sec --
				lea			edx, [esi + 2];								 // hoge051 = (+2) + pointer wblurxp(esi)
				vmovdqu		ymm3, YMMWORD PTR[edx];				 		 // load hoge052(ymm3) top y__plane[hoge051] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge053(ymm3) = hoge052(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm0, ymm3, ymm0;							 // hoge054(ymm0) = hoge053(ymm3) + hoge024(ymm0)

				vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 2];			 // load hoge060(ymm3) top cb_plane[hoge051 + zpitch(ecx) * 2] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge061(ymm3) = hoge060(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm1, ymm3, ymm1;							 // hoge062(ymm1) = hoge061(ymm3) + hoge032(ymm1)

				vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 4];			 // load hoge070(ymm3) top cr_plane[hoge051 + zpitch(ecx) * 4] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge071(ymm3) = hoge070(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm2, ymm3, ymm2;							 // hoge072(ymm2) = hoge071(ymm3) + hoge042(ymm2)


				// -- store dest --
				vmovdqa		YMMWORD PTR[edi], ymm0;						// store y__plane(ymm0)
				vmovdqa		YMMWORD PTR[edi + ecx * 2], ymm1;			// store cb_plane(ymm1)
				vmovdqa		YMMWORD PTR[edi + ecx * 4], ymm1;			// store cr_plane(ymm2)

				pop			edx;
				pop			eax;
				pop			ecx;
				pop			edi;
				pop			esi;
			}
			x += 16;
			workp += 16;
			wblurxp += 16;
		}

		//   todo:ブラー対象非SIMD領域
		for (; x < fpip->w - 2; x++) {
			wblurxp[y__plane] =
				((BLUR_CENTER_MUL * (int)workp[y__plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[y__plane - 1]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[y__plane + 1]) >> 16);
			wblurxp[cb_plane] =
				((BLUR_CENTER_MUL * (int)workp[cb_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cb_plane - 1]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cb_plane + 1]) >> 16);
			wblurxp[cr_plane] =
				((BLUR_CENTER_MUL * (int)workp[cr_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cr_plane - 1]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)workp[cr_plane + 1]) >> 16);
			workp++;
			wblurxp++;
		}

		//   todo:ブラー右対象外域
		for (; x < fpip->w; x++) {
			//フィルタがかからない領域 縦の範囲チェックは除外してもよい
			wblurxp[y__plane] = workp[y__plane];
			wblurxp[cb_plane] = workp[cb_plane];
			wblurxp[cr_plane] = workp[cr_plane];
			workp++;
			wblurxp++;
		}
	}

	//========================================
	// ブラー 縦　要SIMD化
	//========================================
	DBG("[%-16s] [thread(%d)] ========== blur_y ==========\n", __FUNCTION__, thread_id);

	simdcount = fpip->w / 16;	//32バイト＝256ビット short型なので2分の1
	simdcount2 = (fpip->w - 18) / 16;	//32バイト＝256ビット short型なので2分の1
	y_start1 = y_start - BLURY_MARGIN_Y;	// 縦ブラーマージン 
	y_end1 = y_end + BLURY_MARGIN_Y;	// 縦ブラーマージン 
	if (y_start1 <  0) y_start1 = 0;
	if (y_end1 >= fpip->h) y_end1 = fpip->h;
	for (int y = y_start1; y < y_end1; y++) {
		// 処理対象のポインタの計算
		short* wblurxp = wblurx + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		short* wbluryp = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;

		// todo:上下ブラー対象外領域
		if (y <= 1 || y >= fpip->h - 2) {
			int x = 0;
			// -- simd --
			for (int i = 0; i < simdcount; i++) {
				_asm {
					push				edi;
					push				esi;
					push				ecx;

					// -- load pointer --
					mov					edi, DWORD PTR[wblurxp];
					mov					esi, DWORD PTR[wbluryp];

					// -- --
					mov					ecx, DWORD PTR[wpitch];

					// -- copy y__plane --
					vmovdqa             ymm0, YMMWORD PTR[esi];
					vmovdqa             YMMWORD PTR[edi], ymm0;

					// -- copy cb_plane --
					vmovdqa             ymm0, YMMWORD PTR[esi + ecx * 2];
					vmovdqa             YMMWORD PTR[edi + ecx * 2], ymm0;

					// -- copy cr_plane --
					vmovdqa             ymm0, YMMWORD PTR[esi + ecx * 4];
					vmovdqa             YMMWORD PTR[edi + ecx * 4], ymm0;

					pop					ecx;
					pop					esi;
					pop					edi;
				}
				x += 16;
				wblurxp += 16;
				wbluryp += 16;
			}
			// -- 非simd --
			for (; x < fpip->w; x++) {
				//フィルタがかからない領域 縦の範囲チェックは除外してもよい
				wbluryp[y__plane] = wblurxp[y__plane];
				wbluryp[cb_plane] = wblurxp[cb_plane];
				wbluryp[cr_plane] = wblurxp[cr_plane];
				wblurxp++;
				wbluryp++;
			}
			continue;
		}

		// todo:中央
		int x = 0;
		//   todo:ブラー左対象外域
		for (; x < 2; x++) {
			//フィルタがかからない領域 縦の範囲チェックは除外してもよい
			wbluryp[y__plane] = wblurxp[y__plane];
			wbluryp[cb_plane] = wblurxp[cb_plane];
			wbluryp[cr_plane] = wblurxp[cr_plane];
			wblurxp++;
			wbluryp++;
		}

		//   todo:ブラー対象非SIMD領域
		for (; x < 16; x++) {
			wbluryp[y__plane] =
				((BLUR_CENTER_MUL * (int)wblurxp[y__plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[-fpip->w + y__plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[+fpip->w + y__plane]) >> 16);
			wbluryp[cb_plane] =
				((BLUR_CENTER_MUL * (int)wblurxp[cb_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[-fpip->w + cb_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[+fpip->w + cb_plane]) >> 16);
			wbluryp[cr_plane] =
				((BLUR_CENTER_MUL * (int)wblurxp[cr_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[-fpip->w + cr_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[+fpip->w + cr_plane]) >> 16);
			wblurxp++;
			wbluryp++;
		}

		//   todo:ブラー対象SIMD領域
		for (int i = 0; i < simdcount2; i++) {
			_asm {
				push		esi;
				push		edi;
				push		ecx;
				push		eax;
				push		edx;

				// -- load pointer --
				mov			esi, DWORD PTR[wblurxp];		// load hoge001(esi) wblurxp pointer
				mov			edi, DWORD PTR[wbluryp];		// load hoge002(edi) wbluryp pointer

				// -- load --
				mov			ecx, DWORD PTR[zpitch];			// load hoge003(ecx) zpitch
				mov			eax, DWORD PTR[wpitch];			// load hoge003(eax) wpitch

				
				// -- load center value sec --
				vmovdqa		ymm0, YMMWORD PTR[esi];						// load hoge010(ymm0) center y__plane value
				vpmulhw		ymm0, ymm0, ymm6;							// hoge011(ymm0) = hoge010(ymm0) * blur_center_mul_mm(ymm6) high
				vmovdqa		ymm1, YMMWORD PTR[esi + ecx * 2];			// load hoge012(ymm1) center cb_plane[+ecx * 2] value
				vpmulhw		ymm1, ymm1, ymm6;							// hoge013(ymm1) = hoge012(ymm1) * blur_center_mul_mm(ymm6) high
				vmovdqa		ymm2, YMMWORD PTR[esi + ecx * 4];			// load hoge014(ymm2) center cr_plane[+ecx * 4] value
				vpmulhw		ymm2, ymm2, ymm6;							// hoge015(ymm2) = hoge012(ymm2) * blur_center_mul_mm(ymm6) high


				// -- load top y__plane value sec --
				mov			edx, eax;									// hoge020 = wpitch(eax)
				neg			edx;										// hoge021 = hoge020 * -1
				add			edx, edx;									// hoge021 = hoge021 + hoge021
				lea			edx, [esi + edx];							// hoge021 = hoge021 + pointer wblurxp(esi)
				vmovdqa		ymm3, YMMWORD PTR[edx];				 		 // load hoge022(ymm3) top y__plane[hoge021] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge023(ymm3) = hoge022(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm0, ymm3, ymm0;							 // hoge024(ymm0) = hoge023(ymm3) + hoge011(ymm0)

				vmovdqa		ymm3, YMMWORD PTR[edx + ecx * 2];			 // load hoge030(ymm3) top cb_plane[hoge021 + zpitch(ecx) * 2] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge031(ymm3) = hoge030(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm1, ymm3, ymm1;							 // hoge032(ymm1) = hoge031(ymm3) + hoge013(ymm1)

				vmovdqa		ymm3, YMMWORD PTR[edx + ecx * 4];			 // load hoge040(ymm3) top cr_plane[hoge021 + zpitch(ecx) * 4] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge041(ymm3) = hoge040(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm2, ymm3, ymm2;							 // hoge042(ymm2) = hoge041(ymm3) + hoge015(ymm2)


				// -- load bottom y__plane value sec --
				mov			edx, eax;									// hoge050 = wpitch(eax)
				add			edx, edx;									// hoge051 = hoge050 + hoge050
				lea			edx, [esi + edx];							// hoge051 = hoge051 + pointer wblurxp(esi)
				vmovdqa		ymm3, YMMWORD PTR[edx];				 		 // load hoge052(ymm3) top y__plane[hoge051] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge053(ymm3) = hoge052(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm0, ymm3, ymm0;							 // hoge054(ymm0) = hoge053(ymm3) + hoge024(ymm0)

				vmovdqa		ymm3, YMMWORD PTR[edx + ecx * 2];			 // load hoge060(ymm3) top cb_plane[hoge051 + zpitch(ecx) * 2] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge062(ymm3) = hoge060(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm1, ymm3, ymm1;							 // hoge062(ymm1) = hoge061(ymm3) + hoge032(ymm1)

				vmovdqa		ymm3, YMMWORD PTR[edx + ecx * 4];			 // load hoge070(ymm3) top cr_plane[hoge051 + zpitch(ecx) * 4] value
				vpmulhw		ymm3, ymm3, ymm7;							 // hoge071(ymm3) = hoge070(ymm3) * blur_another_mul_mm(ymm7) high
				vpaddsw		ymm2, ymm3, ymm2;							 // hoge072(ymm2) = hoge071(ymm3) + hoge042(ymm2)


				// -- store dest --
				vmovdqa		YMMWORD PTR[edi], ymm0;						// store y__plane(ymm0)
				vmovdqa		YMMWORD PTR[edi + ecx * 2], ymm1;			// store cb_plane(ymm1)
				vmovdqa		YMMWORD PTR[edi + ecx * 4], ymm1;			// store cr_plane(ymm2)

				pop			edx;
				pop			eax;
				pop			ecx;
				pop			edi;
				pop			esi;
			}
			x += 16;
			wblurxp += 16;
			wbluryp += 16;
		}

		//   todo:ブラー対象非SIMD領域
		for (int x = 0; x < fpip->w - 2; x++) {
			wbluryp[y__plane] =
				((BLUR_CENTER_MUL * (int)wblurxp[y__plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[-fpip->w + y__plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[+fpip->w + y__plane]) >> 16);
			wbluryp[cb_plane] =
				((BLUR_CENTER_MUL * (int)wblurxp[cb_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[-fpip->w + cb_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[+fpip->w + cb_plane]) >> 16);
			wbluryp[cr_plane] =
				((BLUR_CENTER_MUL * (int)wblurxp[cr_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[-fpip->w + cr_plane]) >> 16) +
				((BLUR_ANOTHER_MUL * (int)wblurxp[+fpip->w + cr_plane]) >> 16);
			wblurxp++;
			wbluryp++;
		}

		//   todo:ブラー右対象外域
		for (; x < fpip->w; x++) {
			//フィルタがかからない領域 縦の範囲チェックは除外してもよい
			wbluryp[y__plane] = wblurxp[y__plane];
			wbluryp[cb_plane] = wblurxp[cb_plane];
			wbluryp[cr_plane] = wblurxp[cr_plane];
			wblurxp++;
			wbluryp++;
		}

	}

	//========================================
	// マーキング処理 SIMD化不能
	//========================================
	DBG("[%-16s] [thread(%d)] ========== marking ==========\n", __FUNCTION__, thread_id);
//	DBG("[%-16s] threshold   :%d\n", __FUNCTION__, param.threshold);
	unsigned short true_bit_pattern = MSHARPEN_TRUE;

	simdcount = fpip->w / 16;	//32バイト＝256ビット short型なので2分の1
	simdcount2 = (fpip->w - 18) / 16;	//32バイト＝256ビット short型なので2分の1
	//short threshold_mm[16];
	//short true_bit_pattern_mm[16];
	_asm {
		/*
		vpbroadcastw		ymm0, WORD PTR[param.threshold];
		vmovdqu             YMMWORD PTR[threshold_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[true_bit_pattern];
		vmovdqu             YMMWORD PTR[true_bit_pattern_mm], ymm0;
		*/
		vmovdqu             ymm4, YMMWORD PTR[threshold_mm];
	}
	for (int y = y_start; y < y_end; y++) {
		// 処理対象のポインタの計算
		short	*wbluryp = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;	//ブラー
		unsigned short	*wmaskp = wmask + (y - y_start) * wpitch;	//マスク
		int x = 0;
		// フィルタがかからない領域　上下
		if (y <= 1 || y >= fpip->h - 2) {
			/*
			// -- SIMD領域 ---
			for (int i = 0 ; i < simdcount; i++) {
				_asm {
					push				edi;
					mov					edi, DWORD PTR[wmaskp];
					vpxor				ymm0, ymm0, ymm0;
					vmovdqu             YMMWORD PTR[edi], ymm0;
					pop					edi;
				}
				x += 16;
				wmaskp += 16;
				wbluryp += 16;
			}
			// -- 非SIMD領域 ---
			for (; x < fpip->w; x++) {
				*wmaskp = MSHARPEN_FALSE;
				wmaskp++;
				wbluryp++;
			}
			*/
			continue;
		}

		//フィルタがかからない領域
		for (; x < 2; x++) {
			*wmaskp = MSHARPEN_FALSE;
			wmaskp++;
			wbluryp++;
		}
		//中央
		// -- 非SIMD領域 --
		for (; x < 16; x++) {
			*wmaskp = MSHARPEN_FALSE;
			// 斜めを比較(左上,中央)
			if (is_threshold_avx2_over(wbluryp - wpitch - 1, wbluryp, core_param)) {
				*wmaskp = MSHARPEN_TRUE;
				// 斜めを比較(左下,中央)
			}
			else if (is_threshold_avx2_over(wbluryp + wpitch - 1, wbluryp, core_param)) {
				*wmaskp = MSHARPEN_TRUE;
			}
			else if (param.is_high_quality) {
				// 縦を比較(上,中央)
				if (is_threshold_avx2_over(wbluryp - wpitch, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 横を比較(左,中央)
				}
				else if (is_threshold_avx2_over(wbluryp - 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 縦を比較(下,中央)
				}
				else if (is_threshold_avx2_over(wbluryp + wpitch, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 斜め比較(右下,中央)
				}
				else if (is_threshold_avx2_over(wbluryp + wpitch + 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 横を比較(右,中央)
				}
				else if (is_threshold_avx2_over(wbluryp + 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 斜め比較(右上,中央)
				}
				else if (is_threshold_avx2_over(wbluryp - wpitch + 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
				}
			}
			wmaskp++;
			wbluryp++;
		}

		// -- SIMD領域 --
		if (!param.is_high_quality) {
			for (int i = 0; i < simdcount2; i++) {
				_asm {
					push		esi;
					push		edi;
					push		ecx;
					push		eax;
					push		edx;

					// ============================
					// =    load pointer          =
					// ============================
					mov			esi, DWORD PTR[wbluryp];
					mov			edi, DWORD PTR[wmaskp];

					// ============================
					// -- --
					// ============================
					mov			ecx, DWORD PTR[zpitch];
					mov			eax, DWORD PTR[wpitch];

					// ============================
					// -- load center pixel --
					// ============================
					vmovdqa		ymm0, YMMWORD PTR[esi];						// hoge001 = wbluryp[y__plane]
					vmovdqa		ymm1, YMMWORD PTR[esi+ecx*2];				// hoge002 = wbluryp[cb_plane]
					vmovdqa		ymm2, YMMWORD PTR[esi+ecx*4];				// hoge003 = wbluryp[cr_plane]

					// ============================
					// 斜めを比較(左上,中央)
					// ============================
					mov			edx, eax;									// hoge010 = wpitch
					neg			edx;										// hoge011 = hoge010 * -1
					sub			edx, 1;										// hoge012 = hoge011 - 1
					add			edx, edx;									// hoge013 = hoge012 * sizeof(short)
					lea			edx, [esi + edx];							// hoge014 = pointer wbluryp[y__plane] + hoge013

					// -- cmp threshold_mm --
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge9  = *hoge8 [y__plane]
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					//vpcmpgtw	ymm6, ymm6, YMMWORD PTR[threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpcmpgtw	ymm6, ymm6, ymm4;							// hoge22 = (hoge20 >= [threshold_mm])
					//vpand		ymm7, ymm6, YMMWORD PTR[true_bit_pattern_mm];// 
					vmovdqa		ymm7, ymm6;
					//
					vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 2];				// hoge10 = *hoge8 [cb_plane]
					vpsubsw		ymm6, ymm1, ymm3;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					//vpcmpgtw	ymm6, ymm6, YMMWORD PTR[threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpcmpgtw	ymm6, ymm6, ymm4;							// hoge22 = (hoge20 >= [threshold_mm])
					//vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					//vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge22)
					vpor		ymm7, ymm6, ymm7;
					//
					vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 4];				// hoge11 = *hoge8 [cr_plane]
					vpsubsw		ymm6, ymm2, ymm3;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					//vpcmpgtw	ymm6, ymm6, YMMWORD PTR[threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpcmpgtw	ymm6, ymm6, ymm4;							// hoge22 = (hoge20 >= [threshold_mm])
					//vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					//vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)
					vpor		ymm7, ymm6, ymm7;

					// ============================
					// 斜めを比較(左下,中央)
					// ============================
					mov			edx, eax;									// hoge40 = wpitch
					sub			edx, 1;										// hoge41 = hoge40 - 1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42

					// -- cmp threshold_mm --
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					//vpcmpgtw	ymm6, ymm6, YMMWORD PTR[threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpcmpgtw	ymm6, ymm6, ymm4;							// hoge22 = (hoge20 >= [threshold_mm])
					//vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					//vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)
					vpor		ymm7, ymm6, ymm7;

					vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vpsubsw		ymm6, ymm1, ymm3;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					//vpcmpgtw	ymm6, ymm6, YMMWORD PTR[threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpcmpgtw	ymm6, ymm6, ymm4;							// hoge22 = (hoge20 >= [threshold_mm])
					//vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					//vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)
					vpor		ymm7, ymm6, ymm7;

					vmovdqu		ymm3, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]
					vpsubsw		ymm6, ymm2, ymm3;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					//vpcmpgtw	ymm6, ymm6, YMMWORD PTR[threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpcmpgtw	ymm6, ymm6, ymm4;							// hoge22 = (hoge20 >= [threshold_mm])
					//vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					//vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)
					vpor		ymm7, ymm6, ymm7;

					// ============================
					// ============================
					vmovdqa	    YMMWORD PTR[edi], ymm7;

					pop			edx;
					pop			eax;
					pop			ecx;
					pop			edi;
					pop			esi;
				}
				x += 16;
				wmaskp += 16;
				wbluryp += 16;
			}
		} else {
			for (int i = 0; i < simdcount2; i++) {
				_asm {
					push		esi;
					push		edi;
					push		ecx;
					push		eax;
					push		edx;

					// -- load pointer --
					mov			esi, DWORD PTR[wbluryp];
					mov			edi, DWORD PTR[wmaskp];
					// -- --
					mov			ecx, DWORD PTR[zpitch];
					mov			eax, DWORD PTR[wpitch];

					// -- load center pixel --
					vmovdqa		ymm0, YMMWORD PTR[esi];						// hoge1 = wbluryp[y__plane]
					vmovdqa		ymm1, YMMWORD PTR[esi + ecx * 2];				// hoge2 = wbluryp[cb_plane]
					vmovdqa		ymm2, YMMWORD PTR[esi + ecx * 4];				// hoge3 = wbluryp[cr_plane]

					// 斜めを比較(左上,中央)
					// -- - wpitch - 1 -- 
					mov			edx, eax;									// hoge4 = wpitch
					neg			edx;										// hoge5 = hoge4 * -1
					sub			edx, 1;										// hoge6 = hoge5 - 1
					add			edx, edx;									// hoge7 = hoge6 * sizeof(short)
					lea			edx, [esi + edx];							// hoge8 = pointer wbluryp[y__plane] + hoge7
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge9  = *hoge8 [y__plane]
					vmovdqu		ymm4, YMMWORD PTR[edx + ecx * 2];				// hoge10 = *hoge8 [cb_plane]
					vmovdqu		ymm5, YMMWORD PTR[edx + ecx * 4];				// hoge11 = *hoge8 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm7, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge22)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 斜めを比較(左下,中央)
					// -- + wpitch - 1 -- 
					mov			edx, eax;									// hoge40 = wpitch
					sub			edx, 1;										// hoge41 = hoge40 - 1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqu		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqu		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 縦を比較(上,中央) 
					// -- - wpitch -- 
					mov			edx, eax;									// hoge40 = wpitch
					neg			edx;										// hoge40 = hoge40 * -1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqa		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqa		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqa		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 横を比較(左,中央)
					// -- - 1 --
					mov			edx, -1;									// hoge40 = -1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqu		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqu		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 縦を比較(下,中央)
					// -- + wpitch --
					mov			edx, eax;									// hoge40 = wpitch
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqa		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqa		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqa		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 斜め比較(右下,中央)
					// -- + wpitch + 1 --
					mov			edx, eax;									// hoge40 = wpitch
					add			edx, 1;										// hoge41 = hoge40 + 1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqu		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqu		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 横を比較(右,中央)
					// -- + 1 --
					mov			edx, 1;										// hoge40 = 1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqu		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqu		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					// 斜め比較(右上,中央)
					// -- - wpitch + 1 --
					mov			edx, eax;									// hoge40 = wpitch
					neg			edx;										// hoge40 = hoge40 * -1
					add			edx, 1;										// hoge41 = hoge40 + 1
					add			edx, edx;									// hoge42 = hoge41 * sizeof(short)
					lea			edx, [esi + edx];							// hoge43 = pointer wbluryp[y__plane] + hoge42
					vmovdqu		ymm3, YMMWORD PTR[edx];						// hoge44 = *hoge43 [y__plane]
					vmovdqu		ymm4, YMMWORD PTR[edx + ecx * 2];			// hoge45 = *hoge43 [cb_plane]
					vmovdqu		ymm5, YMMWORD PTR[edx + ecx * 4];			// hoge46 = *hoge43 [cr_plane]

					// -- cmp threshold_mm --
					vpsubsw		ymm6, ymm0, ymm3;							// hoge20 = hoge1 - hoge9
					vpabsw		ymm6, ymm6;									// hoge21 = abs(hoge20)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge22 = (hoge20 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];		// 
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge225 = max(hoge22, hoge30)

					vpsubsw		ymm6, ymm1, ymm4;							// hoge23 = hoge2 - hoge10
					vpabsw		ymm6, ymm6;									// hoge24 = abs(hoge23)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge25 = (hoge24 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge26 = max(hoge25, hoge225)

					vpsubsw		ymm6, ymm2, ymm5;							// hoge27 = hoge3 - hoge11
					vpabsw		ymm6, ymm6;									// hoge28 = abs(hoge27)
					vpcmpgtw	ymm6, ymm6, YMMWORD PTR[     threshold_mm];		// hoge29 = (hoge28 >= [threshold_mm])
					vpand		ymm6, ymm6, YMMWORD PTR[     true_bit_pattern_mm];
					vpmaxuw		ymm7, ymm6, ymm7;							// hoge30 = max(hoge29, hoge26)

					vmovdqa	    YMMWORD PTR[edi], ymm7;

					pop			edx;
					pop			eax;
					pop			ecx;
					pop			edi;
					pop			esi;
				}
				x += 16;
				wmaskp += 16;
				wbluryp += 16;
			}
		}

		// -- 非SIMD領域 --
		for (; x < fpip->w - 2; x++) {
			*wmaskp = MSHARPEN_FALSE;
			// 斜めを比較(左上,中央)
			if (is_threshold_avx2_over(wbluryp - wpitch - 1, wbluryp, core_param)) {
				*wmaskp = MSHARPEN_TRUE;
				// 斜めを比較(左下,中央)
			}
			else if (is_threshold_avx2_over(wbluryp + wpitch - 1, wbluryp, core_param)) {
				*wmaskp = MSHARPEN_TRUE;
			}
			else if (param.is_high_quality) {
				// 縦を比較(上,中央)
				if (is_threshold_avx2_over(wbluryp - wpitch, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 横を比較(左,中央)
				}
				else if (is_threshold_avx2_over(wbluryp - 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 縦を比較(下,中央)
				}
				else if (is_threshold_avx2_over(wbluryp + wpitch, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 斜め比較(右下,中央)
				}
				else if (is_threshold_avx2_over(wbluryp + wpitch + 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 横を比較(右,中央)
				}
				else if (is_threshold_avx2_over(wbluryp + 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
					// 斜め比較(右上,中央)
				}
				else if (is_threshold_avx2_over(wbluryp - wpitch + 1, wbluryp, core_param)) {
					*wmaskp = MSHARPEN_TRUE;
				}
			}
			wmaskp++;
			wbluryp++;
		}

		//フィルタがかからない領域
		for (; x < fpip->w; x++) {
			*wmaskp = MSHARPEN_FALSE;
			wmaskp++;
			wbluryp++;
		}
	}

	//========================================
	// マスク結果の返却 実エンコでは使用されない想定なのでSIMD化しない
	//========================================
	if (param.is_mask) {
		for (int y = y_start; y < y_end; y++) {
			// 処理対象のポインタの計算
			PIXEL_YC *ycp_temp = fpip->ycp_temp + y            * max_w;	// テンポラリ領域
			unsigned short     *wmaskp = wmask + (y - y_start) * wpitch;	//マスク

			// 横方向ループ
			for (int x = 0; x < fpip->w; x++) {
				// 色差はフィルタの対象外
				ycp_temp->cb = 0;
				ycp_temp->cr = 0;
				// 輝度に対する処理
				if (*wmaskp == MSHARPEN_TRUE) {
					ycp_temp->y = 3072;
				}
				else {
					ycp_temp->y = 256;
				}
				//
				ycp_temp++;
				wmaskp++;
			}
		}
		return;
	}

	//========================================
	// 結果の適用　SIMD化完了
	//========================================
	DBG("[%-16s] [thread(%d)] ========== detect ==========\n", __FUNCTION__, thread_id);
	simdcount = fpip->w / 16;	//32バイト＝256ビット short型なので2分の1
	short detect_max = 0;
	short detect_min = 4096;
	/*
	DBG("[%-16s] simdcount   :%d\n", __FUNCTION__, simdcount);
	DBG("[%-16s] strength    :%d\n", __FUNCTION__, param.strength);
	DBG("[%-16s] invstrength :%d\n", __FUNCTION__, param.invstrength);
	DBG("[%-16s] detect_max  :%d\n", __FUNCTION__, detect_max);
	DBG("[%-16s] detect_min  :%d\n", __FUNCTION__, detect_min);
	DBG("[%-16s] true_bit_pattern  :%d\n", __FUNCTION__, true_bit_pattern);
	*/
	//
	/*
	short strength_mm[16];
	short invstrength_mm[16];
	short detect_max_mm[16];
	short detect_min_mm[16];
	*/
	_asm {
		/*
		vpbroadcastw		ymm0, WORD PTR[param.strength];
		vmovdqu             YMMWORD PTR[strength_mm], ymm0;
		//vmovdqu				ymm2, ymm0;
		vpbroadcastw		ymm0, WORD PTR[param.invstrength];
		vmovdqu             YMMWORD PTR[invstrength_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[detect_max];
		vmovdqu             YMMWORD PTR[detect_max_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[detect_min];
		vmovdqu             YMMWORD PTR[detect_min_mm], ymm0;
		*/
		vmovdqu             ymm6, YMMWORD PTR[true_bit_pattern_mm];
		vmovdqu             ymm5, YMMWORD PTR[strength_mm];
		vmovdqu             ymm7, YMMWORD PTR[invstrength_mm];
	}
	/*
	DBG("[%-16s] strength_mm    :%d\n", __FUNCTION__, strength_mm[15]);
	DBG("[%-16s] invstrength_mm :%d\n", __FUNCTION__, invstrength_mm[15]);
	DBG("[%-16s] detect_max_mm  :%d\n", __FUNCTION__, detect_max_mm[15]);
	DBG("[%-16s] detect_min_mm  :%d\n", __FUNCTION__, detect_min_mm[15]);
	DBG("[%-16s] true_bit_pattern_mm  m  :%d\n", __FUNCTION__, true_bit_pattern_mm[15]);
	*/

	for (int y = y_start; y < y_end; y++) {
		// 処理対象のポインタの計算
		short		*workp = work + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		short		*wbluryp = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;	//ブラー
		unsigned short *wmaskp = wmask + (y - y_start) * wpitch;						//マスク

		// todo:中央
		int x = 0;
		///*
		//   todo:適用SIMD領域
		for (int i = 0; i < simdcount; i++) {
			// ポインタインク
			_asm {
				// ymm0 workp
				// ymm1 wmaskp
				// ymm2 wbluryp
				// ymm3 
				// ymm4 
				// ymm5 
				// ymm6 
				// ymm7 
				push		esi;
				push		ecx;
				push		eax;

				// ============================
				// =    pointer load sec      =
				// ============================
				mov			esi, DWORD PTR[workp];
				mov			ecx, DWORD PTR[wmaskp];
				mov			eax, DWORD PTR[wbluryp];

				// ============================
				// =    load sec              =
				// ============================
				vmovdqa     ymm0, YMMWORD PTR[esi];                     //  load 16pixel workp hoge000
				vmovdqa     ymm1, YMMWORD PTR[ecx];                     //  load 16pixel wmaskp

				// ============================
				// =    detect == TRUE sec    =
				// ============================
				// -- calc y --
				vmovdqa     ymm3, YMMWORD PTR[eax];                     //  load 16pixel wbluryp hoge001

				vmovdqu		YMMWORD PTR[test1], ymm3;					// これを外すとバグる　原因不明

				vpsllw      ymm2, ymm3, 1;                              //  hoge010 = hoge001 * 3  = hoge010 = hoge001(wbluryp[y__plane]) << 1
				vpaddw      ymm2, ymm2, ymm3;                           //                           hoge010 = hoge010 + hoge001(wbluryp[y__plane])
				vpsllw      ymm3, ymm0, 2;                              //  hoge011 = hoge000(workp[y__plane]:ymm0) * 4  =  hoge002 = hoge000(workp[y__plane]:ymm0) << 2
				vpsubsw     ymm2, ymm3, ymm2;                           //  hoge012 = hoge011 - hoge010
				vpminsw     ymm2, ymm2, YMMWORD PTR[detect_min_mm];//  hoge013 = min(hoge012, 4096)

				vpmaxsw     ymm2, ymm2, YMMWORD PTR[detect_max_mm];//  y = max(hoge013, 0)
				//vpxor		ymm3, ymm3, ymm3;
				//vpmaxsw     ymm2, ymm2, ymm3;							//  y = max(hoge013, 0)

				// -- hoge032  = ( param.strength * y )>> 8 --
				vpmulhw     ymm3, ymm2, ymm5;                           //  hoge020  = param.strength * y high
				vpsllw      ymm3, ymm3, 8;								//  hoge021 = hoge020 (param.strength * y high) << 8
				vpmullw     ymm4, ymm2, ymm5;                           //  hoge030 = param.strength * y low
				vpsrlw      ymm4, ymm4, 8;								//	hoge031 = hoge030(param.strength * y low)  >> 8
				vpor		ymm2, ymm3, ymm4;							//  hoge032 = hoge021 or hoge031

				// -- hoge052  = ( param.invstrength * workp[y__plane] )>> 8
				vpmulhw     ymm3, ymm0, ymm7;                           //  hoge040 = param.invstrength * workp[y__plane] high
				vpsllw      ymm3, ymm3, 8;								//  hoge041 = hoge040(param.invstrength * workp[y__plane] high) << 8
				vpmullw     ymm4, ymm0, ymm7;                           //  hoge050 = param.invstrength * workp[y__plane] low
				vpsrlw      ymm4, ymm4, 8;								//	hoge051 = hoge050(param.invstrength * workp[y__plane] low)  >> 8
				vpor		ymm3, ymm3, ymm4;							//  hoge052 = hoge041 or hoge051

				// -- hoge061 = hoge032 + hoge052
				vpaddsw     ymm2, ymm2, ymm3;                           //  hoge061 = hoge032 + hoge052
				vpand	    ymm2, ymm2, ymm1;                           //  hoge062 = hoge061 and ymm1(*wmaskp)(0x0000 or 0xffff)

				// ============================
				// =    detect == FALSE sec   =
				// ============================
				vpxor	    ymm3, ymm6, ymm1;                           //  hoge100 = ymm6(true_bit_pattern) xor ymm1(*wmaskp)
				vpand	    ymm3, ymm0, ymm3;                           //  hoge101 = ymm0 and hoge100(0x0000 or 0xffff)

				// ============================
				// =    detect marge          =
				// ============================
				vpaddsw     ymm2, ymm2, ymm3;                           //  hoge200 = hoge062 + hoge101

				// ============================
				// =    strore hoge9          =
				// ============================
				vmovdqa     YMMWORD PTR[esi], ymm2;                     //  strore hoge200

				pop			eax;
				pop			ecx;
				pop			esi;
			}
			/*
			// test
			for (int j = 0; j < 8; j++) {
				if (wbluryp[j] != test1[j]) 
					DBG("dif %d - %d.", wbluryp[j], test1[j]);
				*//*
				if (wmaskp[j] == MSHARPEN_TRUE) {
					int y = max(min(workp[j + y__plane] * 4 - wbluryp[j + y__plane] * 3, 4096), 0);
					if (y != test1[j]) {
						DBG("dif %d - %d.", y, test1[j]);
					}
					//				workp[y__plane] = 
					//					(param.strength * y) >> 8 + 
					//					(param.invstrength * workp[y__plane]) >> 8;
				}
				*//*
			}
			*/

			x += 16;
			workp += 16;
			wbluryp += 16;
			wmaskp += 16;
		}

		//*/
		//   todo:適用非SIMD領域
		// 横方向ループ
		for (; x < fpip->w; x++) {

			// 輝度に対する処理
			if (*wmaskp == MSHARPEN_TRUE) {
				int y = max(min(workp[y__plane] * 4 - wbluryp[y__plane] * 3, 4096), 0);
				workp[y__plane] = 
					(param.strength * y) >> 8 + 
					(param.invstrength * workp[y__plane]) >> 8;
			}

			// ポインタインク
			workp++;
			wbluryp++;
			wmaskp++;
		}
	}

	//========================================
	// プレーンコピー　要SIMD化
	//========================================
	DBG("[%-16s] [thread(%d)] ========== copy_temp ==========\n", __FUNCTION__, thread_id);
	simdcount = fpip->w / 32;	//8*4*2=64バイト＝512ビット short型なので2分の1

	y_start1 = y_start;	// 縦ブラーマージン 
	y_end1 = y_end;	// 縦ブラーマージン 
	for (int y = y_start1; y < y_end1; y++) {
		// 処理対象のポインタの計算
		short* workp = work + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		PIXEL_YC* ycp_temp = fpip->ycp_temp + y * max_w;		// 画像データ
		int x = 0;
		//   todo:SIMD領域
		/*
		for (int i = 0; i < simdcount; i++) {
			_asm {
				push		esi;
				push		edi;
				push		edx;
				push		ecx;
				push		eax;

				// -- load pointer --
				mov			esi, DWORD PTR[workp];			// load hoge001(esi) workp pointer
				mov			edi, DWORD PTR[ycp_temp];		// load hoge002(edi) ycp_temp pointer

				// -- load --
				mov			eax, DWORD PTR[zpitch];			// load hoge003(eax) zpitch

				// -- y__plane load sec --
				vmovdqa		xmm0, XMMWORD PTR[esi];				    		// load : Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vmovdqa		xmm1, XMMWORD PTR[esi + 16];					// load : Y08,Y09|Y10,Y11|Y12,Y13|Y14,Y15
				vmovdqa		xmm2, XMMWORD PTR[esi + 32];					// load : Y16,Y17|Y18,Y19|Y20,Y21|Y22,Y23
				vmovdqa		xmm3, XMMWORD PTR[esi + 48];					// load : Y24,Y25|Y26,Y27|Y28,Y29|Y30,Y31

				// -- y__plane store sec --
				xor			edx, edx;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 0;			// Y00,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 0;			// Y08,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 0;			// Y16,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 0;			// Y24,---|---,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 1;			// Y00,Y01|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 1;			// Y08,Y09|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 1;			// Y16,Y17|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 1;			// Y24,Y25|---,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 2;			// Y00,Y01|Y02,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 2;			// Y08,Y09|Y10,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 2;			// Y16,Y17|Y18,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 2;			// Y24,Y25|Y26,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 3;			// Y00,Y01|Y02,Y03|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 3;			// Y08,Y09|Y10,Y11|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 3;			// Y16,Y17|Y18,Y19|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 3;			// Y24,Y25|Y26,Y27|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 4;			// Y00,Y01|Y02,Y03|Y04,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 4;			// Y08,Y09|Y10,Y11|Y12,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 4;			// Y16,Y17|Y18,Y19|Y20,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 4;			// Y24,Y25|Y26,Y27|Y28,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 5;			// Y00,Y01|Y02,Y03|Y04,Y05|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 5;			// Y08,Y09|Y10,Y11|Y12,Y13|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 5;			// Y16,Y17|Y18,Y19|Y20,Y21|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 5;			// Y24,Y25|Y26,Y27|Y28,Y29|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 6;			// Y00,Y01|Y02,Y03|Y04,Y05|Y06,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 6;			// Y08,Y09|Y10,Y11|Y12,Y13|Y14,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 6;			// Y16,Y17|Y18,Y19|Y20,Y21|Y22,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 6;			// Y24,Y25|Y26,Y27|Y28,Y29|Y30,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 7;			// Y00,Y01|Y02,Y03|Y04,Y05|Y06,Y07
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 7;			// Y08,Y09|Y10,Y11|Y12,Y13|Y14,Y15
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 7;			// Y16,Y17|Y18,Y19|Y20,Y21|Y22,Y23
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 7;			// Y24,Y25|Y26,Y27|Y28,Y29|Y30,Y31


				// -- cb_plane load sec --
				mov			ecx, eax;
				add			ecx, ecx;
				vmovdqa		xmm0, XMMWORD PTR[esi + ecx];			   		// load : U00,U01|U02,U03|U04,U05|U06,U07
				vmovdqa		xmm1, XMMWORD PTR[esi + ecx + 16];				// load : U08,U09|U10,U11|U12,U13|U14,U15
				vmovdqa		xmm2, XMMWORD PTR[esi + ecx + 32];				// load : U16,U17|U18,U19|U20,U21|U22,U23
				vmovdqa		xmm3, XMMWORD PTR[esi + ecx + 48];				// load : U24,U25|U26,U27|U28,U29|U30,U31

				// -- cb_plane store sec --
				mov			edx, 2;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 0;			// U00,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 0;			// U08,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 0;			// U16,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 0;			// U24,---|---,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 1;			// U00,U01|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 1;			// U08,U09|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 1;			// U16,U17|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 1;			// U24,U25|---,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 2;			// U00,U01|U02,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 2;			// U08,U09|U10,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 2;			// U16,U17|U18,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 2;			// U24,U25|U26,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 3;			// U00,U01|U02,U03|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 3;			// U08,U09|U10,U11|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 3;			// U16,U17|U18,U19|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 3;			// U24,U25|U26,U27|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 4;			// U00,U01|U02,U03|U04,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 4;			// U08,U09|U10,U11|U12,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 4;			// U16,U17|U18,U19|U20,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 4;			// U24,U25|U26,U27|U28,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 5;			// U00,U01|U02,U03|U04,U05|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 5;			// U08,U09|U10,U11|U12,U13|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 5;			// U16,U17|U18,U19|U20,U21|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 5;			// U24,U25|U26,U27|U28,U29|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 6;			// U00,U01|U02,U03|U04,U05|U06,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 6;			// U08,U09|U10,U11|U12,U13|U14,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 6;			// U16,U17|U18,U19|U20,U21|U22,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 6;			// U24,U25|U26,U27|U28,U29|U30,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 7;			// U00,U01|U02,U03|U04,U05|U06,U07
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 7;			// U08,U09|U10,U11|U12,U13|U14,U15
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 7;			// U16,U17|U18,U19|U20,U21|U22,U23
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 7;			// U24,U25|U26,U27|U28,U29|U30,U31


				// -- cr_plane load sec --
				mov			ecx, eax;
				shl			ecx, 2;
				vmovdqa		xmm0, XMMWORD PTR[esi + ecx];			   		// load : V00,V01|V02,V03|V04,V05|V06,V07
				vmovdqa		xmm1, XMMWORD PTR[esi + ecx + 16];				// load : V08,V09|V10,V11|V12,V13|V14,V15
				vmovdqa		xmm2, XMMWORD PTR[esi + ecx + 32];				// load : V16,V17|V18,V19|V20,V21|V22,V23
				vmovdqa		xmm3, XMMWORD PTR[esi + ecx + 48];				// load : V24,V25|V26,V27|V28,V29|V30,V31

				// -- cr_plane store sec --
				mov			edx, 4;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 0;			// V00,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 0;			// V08,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 0;			// V16,---|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 0;			// V24,---|---,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 1;			// V00,V01|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 1;			// V08,V09|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 1;			// V16,V17|---,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 1;			// V24,V25|---,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 2;			// V00,V01|V02,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 2;			// V08,V09|V10,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 2;			// V16,V17|V18,---|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 2;			// V24,V25|V26,---|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 3;			// V00,V01|V02,V03|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 3;			// V08,V09|V10,V11|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 3;			// V16,V17|V18,V19|---,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 3;			// V24,V25|V26,V27|---,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 4;			// V00,V01|V02,V03|V04,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 4;			// V08,V09|V10,V11|V12,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 4;			// V16,V17|V18,V19|V20,---|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 4;			// V24,V25|V26,V27|V28,---|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 5;			// V00,V01|V02,V03|V04,V05|---,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 5;			// V08,V09|V10,V11|V12,V13|---,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 5;			// V16,V17|V18,V19|V20,V21|---,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 5;			// V24,V25|V26,V27|V28,V29|---,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 6;			// V00,V01|V02,V03|V04,V05|V06,---
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 6;			// V08,V09|V10,V11|V12,V13|V14,---
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 6;			// V16,V17|V18,V19|V20,V21|V22,---
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 6;			// V24,V25|V26,V27|V28,V29|V30,---
				add			edx, 6;
				vpextrw		XMMWORD PTR[edi + edx + 0], xmm0, 7;			// V00,V01|V02,V03|V04,V05|V06,V07
				vpextrw		XMMWORD PTR[edi + edx + 48], xmm1, 7;			// V08,V09|V10,V11|V12,V13|V14,V15
				vpextrw		XMMWORD PTR[edi + edx + 96], xmm2, 7;			// V16,V17|V18,V19|V20,V21|V22,V23
				vpextrw		XMMWORD PTR[edi + edx + 144], xmm3, 7;			// V24,V25|V26,V27|V28,V29|V30,V31

				pop			eax;
				pop			ecx;
				pop			edx;
				pop			edi;
				pop			esi;
			}
			x += 32;
			ycp_temp += 32;
			workp += 32;
		}
		*/
		//   todo:非SIMD領域
		for (; x < fpip->w; x++) {
			ycp_temp->y  = workp[y__plane];
			ycp_temp->cb = workp[cb_plane];
			ycp_temp->cr = workp[cr_plane];
			ycp_temp++;
			workp++;
		}
	}


}

//---------------------------------------------------------------------
//		フィルタ処理関数 AVX2前提
//---------------------------------------------------------------------
BOOL cpu_filter_avx2_func(
	_In_ FILTER* fp,
	_In_ FILTER_PROC_INFO* fpip)
{
	DBG("[%-16s] start\n", __FUNCTION__);

	//========================================
	// チェック
	//========================================
	assert(fp != NULL);
	assert(fp->track != NULL);
	assert(fp->check != NULL);
	assert(fpip != NULL);
	assert(fpip->ycp_edit != NULL);
	assert(fpip->ycp_temp != NULL);

	//========================================
	// 実行スレッド数を取得
	//========================================
	int thread_num;
	fp->exfunc->exec_multi_thread_func(msharpen_get_thread_num, (void *)&thread_num, NULL);
	//msharpen_get_thread_num(0, 1, (void *)&thread_num, NULL);
	DBG("[%-16s] msharpen_get_thread_num thread_num %d\n", __FUNCTION__, thread_num);

	//========================================
	// 定数初期化
	//========================================
	MSHARPEN_AVX2_CORE_PARAM fparam;
	short blur_center_mul = 22444;
	short blur_another_mul = 21546;
	short detect_max = 0;
	short detect_min = 4096;
	unsigned short true_bit_pattern = MSHARPEN_TRUE;
	_asm {
		vpbroadcastw		ymm0, WORD PTR[blur_center_mul];
		vmovdqu             YMMWORD PTR[fparam.blur_center_mul_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[blur_another_mul];
		vmovdqu             YMMWORD PTR[fparam.blur_another_mul_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[param.threshold];
		vmovdqu             YMMWORD PTR[fparam.threshold_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[true_bit_pattern];
		vmovdqu             YMMWORD PTR[fparam.true_bit_pattern_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[param.threshold];
		vmovdqu             YMMWORD PTR[fparam.threshold_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[true_bit_pattern];
		vmovdqu             YMMWORD PTR[fparam.true_bit_pattern_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[param.strength];
		vmovdqu             YMMWORD PTR[fparam.strength_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[param.invstrength];
		vmovdqu             YMMWORD PTR[fparam.invstrength_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[detect_max];
		vmovdqu             YMMWORD PTR[fparam.detect_max_mm], ymm0;
		vpbroadcastw		ymm0, WORD PTR[detect_min];
		vmovdqu             YMMWORD PTR[fparam.detect_min_mm], ymm0;
	}

	//========================================
	// ワークメモリの確保
	//========================================
	fparam.thread_step =
		(fpip->h / thread_num) + (fpip->h % thread_num != 0 ? 1 : 0) + 
		BLURX_MARGIN_Y * 2;
	fparam.wpitch = (fpip->w + 31) & (~31);			//2014.08.31 AVX2に向けて補正256bit=32byte
	fparam.zpitch = fparam.thread_step * thread_num * fparam.wpitch;
	fparam.y__plane = fparam.zpitch * 0;
	fparam.cb_plane = fparam.zpitch * 1;
	fparam.cr_plane = fparam.zpitch * 2;
	DBG("[%-16s] thread_step %d\n", __FUNCTION__, fparam.thread_step);
	DBG("[%-16s] wpitch %d\n", __FUNCTION__, fparam.wpitch);
	DBG("[%-16s] zpitch %d\n", __FUNCTION__, fparam.zpitch);
	DBG("[%-16s] y__plane %d\n", __FUNCTION__, fparam.y__plane);
	DBG("[%-16s] cb_plane %d\n", __FUNCTION__, fparam.cb_plane);
	DBG("[%-16s] cr_plane %d\n", __FUNCTION__, fparam.cr_plane);

	short* work = new short[fparam.zpitch * 3 + 16];
	short* wblurx = new short[fparam.zpitch * 3 + 16];
	short* wblury = new short[fparam.zpitch * 3 + 16];
	short* wmask = new short[fparam.zpitch + 16];
	ZeroMemory(work, sizeof(short) * (fparam.zpitch * 3 + 16));
	ZeroMemory(wblurx, sizeof(short) * (fparam.zpitch * 3 + 16));
	ZeroMemory(wblury, sizeof(short) * (fparam.zpitch * 3 + 16));
	ZeroMemory(wmask, sizeof(short) * (fparam.zpitch + 16));
	// 256bit アドレッシング
	fparam.work = (short*)(((int)work + 31) & (~31));
	fparam.wblurx = (short*)(((int)wblurx + 31) & (~31));
	fparam.wblury = (short*)(((int)wblury + 31) & (~31));
	fparam.wmask = (unsigned short*)(((int)wmask + 31) & (~31));

	//========================================
	// フィルターの実行
	//========================================
	DBG("[%-16s] execute msharpen_avx2_core\n", __FUNCTION__);
	fp->exfunc->exec_multi_thread_func(msharpen_avx2_core, (void *)&fparam, (void *)fpip);
	//msharpen_avx2_core(0, 1, (void *)&fparam, (void *)fpip);

	//========================================
	// ワークメモリの後始末
	//========================================
	delete[] work;
	delete[] wblurx;
	delete[] wblury;
	delete[] wmask;

	DBG("[%-16s] end\n", __FUNCTION__);
	return TRUE;
}

#pragma endregion
