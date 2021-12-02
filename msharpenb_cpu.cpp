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

typedef struct tagMSHARPEN_CORE_PARAM
{
	// 共有メモリの設定
	int thread_step;
	// - ブラー作業用
	PIXEL_YC *wblurx;
	// - ブラー作業用2
	PIXEL_YC *wblury;
	// - マスク作業用
	bool     *wmask;
	//2014.09.01 ↓SIMD向けに実相
	// - ブラー作業用横ピッチ
	int		wpitch;		
	//2014.09.01 ↑SIMD向けに実相

} MSHARPEN_CORE_PARAM, *LPMSHARPEN_CORE_PARAM;

#pragma region non simd filter

//---------------------------------------------------------------------
//		インライン関数
//---------------------------------------------------------------------
// 閾値チェック
inline BOOL is_threshold_over(_In_ const PIXEL_YC yca, _In_ const PIXEL_YC ycb) {
	if (abs(yca.y - ycb.y) >= param.threshold) return TRUE;
	if (abs(yca.cb - ycb.cb) >= param.threshold) return TRUE;
	if (abs(yca.cr - ycb.cr) >= param.threshold) return TRUE;
	return FALSE;
}

//---------------------------------------------------------------------
//		フィルタ処理関数 non simdi
//---------------------------------------------------------------------
void msharpen_core( 
	_In_ int thread_id,
	_In_ int thread_num,
	_In_ void *param1,
	_In_ void *param2)
{
	//========================================
	// チェック
	//========================================
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param2;
	assert(fpip != NULL);
	assert(fpip->ycp_edit != NULL);
	assert(fpip->ycp_temp != NULL);

	// ========================================
	// マルチスレッドでの処理分割
	// ========================================
	int y_start = ( fpip->h * thread_id     ) / thread_num;
	int y_end   = ( fpip->h * (thread_id+1) ) / thread_num;

	//========================================
	// 初期化
	//========================================
	int max_w  = fpip->max_w;						// 画像領域のサイズ 横幅
	// ブラー用作業メモリ
//	int wpitch = (fpip->w + 7) & (~7);
	//PIXEL_YC *wblurx = new PIXEL_YC[(y_end - y_start + BLURX_MARGIN_Y * 2) * wpitch];
	//PIXEL_YC *wblury = new PIXEL_YC[(y_end - y_start + BLURY_MARGIN_Y * 2) * wpitch];
	//// マーキング用作業メモリの確保
	//bool      *wmask = new bool    [(y_end - y_start                     ) * wpitch];
	MSHARPEN_CORE_PARAM* core_param = (MSHARPEN_CORE_PARAM*)param1;
	int wpitch = core_param->wpitch;
	PIXEL_YC *wblurx = core_param->wblurx + core_param->thread_step * thread_id * wpitch;
	PIXEL_YC *wblury = core_param->wblury + core_param->thread_step * thread_id * wpitch;
	// マーキング用作業メモリの確保
	bool      *wmask = core_param->wmask  + core_param->thread_step * thread_id * wpitch;
	
	//========================================
	// ブラー 横
	//========================================
	int y_start1 = y_start - BLURX_MARGIN_Y;	// 縦ブラーマージン 
	int y_end1   = y_end   + BLURX_MARGIN_Y;	// 縦ブラーマージン 
	if (y_start1 <  0      ) y_start1 = 0;
	if (y_end1   >= fpip->h) y_end1   = fpip->h;
	for (int y = y_start1 ; y < y_end1 ; y++ ) {
		// 処理対象のポインタの計算
		PIXEL_YC* ycp     = fpip->ycp_edit +  y * max_w;		// 画像データ
		PIXEL_YC* wblurxp = wblurx         + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		for (int x = 0; x < fpip->w; x++ ) {
			if (x <= 1 || x >= fpip->w - 2 || y <= 1 || y >= fpip->h - 2) {
				//フィルタがかからない領域 横の範囲チェックは除外してもよい
				*wblurxp = *ycp;
			} else {
				//横ブラー
				wblurxp->y  = (short)((22444 * (int)ycp[0].y  + 21546 * (int)ycp[-1].y  + 21546 * (int)ycp[+1].y ) >> 16);
				wblurxp->cb = (short)((22444 * (int)ycp[0].cb + 21546 * (int)ycp[-1].cb + 21546 * (int)ycp[+1].cb) >> 16);
				wblurxp->cr = (short)((22444 * (int)ycp[0].cr + 21546 * (int)ycp[-1].cr + 21546 * (int)ycp[+1].cr) >> 16);
			}
			ycp++;
			wblurxp++;
		}
	}

	//========================================
	// ブラー 縦
	//========================================
	y_start1 = y_start - BLURY_MARGIN_Y;	// 縦ブラーマージン 
	y_end1   = y_end   + BLURY_MARGIN_Y;	// 縦ブラーマージン 
	if (y_start1 <  0      ) y_start1 = 0;
	if (y_end1   >= fpip->h) y_end1   = fpip->h;
	for (int y = y_start1 ; y < y_end1 ; y++ ) {
		// 処理対象のポインタの計算
		PIXEL_YC* wblurxp   = wblurx + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		PIXEL_YC* wbluryp   = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;
		for (int x = 0; x < fpip->w; x++ ) {
			if (x <= 1 || x >= fpip->w - 2 || y <= 1 || y >= fpip->h - 2) {
				//フィルタがかからない領域 縦の範囲チェックは除外してもよい
				*wbluryp = *wblurxp;
			} else {
				//横ブラー
				wbluryp->y  = (short)((22444 * (int)wblurxp[0].y  + 21546 * (int)wblurxp[-fpip->w].y  + 21546 * (int)wblurxp[+fpip->w].y ) >> 16);
				wbluryp->cb = (short)((22444 * (int)wblurxp[0].cb + 21546 * (int)wblurxp[-fpip->w].cb + 21546 * (int)wblurxp[+fpip->w].cb) >> 16);
				wbluryp->cr = (short)((22444 * (int)wblurxp[0].cr + 21546 * (int)wblurxp[-fpip->w].cr + 21546 * (int)wblurxp[+fpip->w].cr) >> 16);
			}
			wblurxp++;
			wbluryp++;
		}
	}

	//========================================
	// マーキング処理
	//========================================
	for (int y = y_start ; y < y_end ; y++ ) {
		// 処理対象のポインタの計算
		PIXEL_YC *wbluryp = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;	//ブラー
		bool     *wmaskp  = wmask  + (y - y_start                 ) * wpitch;	//マスク
		for (int x = 0; x < fpip->w; x++ ) {
			if (x <= 1 || x >= fpip->w - 2 || y <= 1 || y >= fpip->h - 2) {
				//フィルタがかからない領域
				*wmaskp = false;
			} else {
				// 斜めを比較(左上,中央)
				if        (is_threshold_over(wbluryp[-fpip->w-1], wbluryp[0])) {
					*wmaskp = true;
				// 斜めを比較(左下,中央)
				} else if (is_threshold_over(wbluryp[+fpip->w-1], wbluryp[0])) {
					*wmaskp = true;
				} else if (param.is_high_quality) {
					// 縦を比較(上,中央)
					if        (is_threshold_over(wbluryp[-fpip->w  ], wbluryp[0])) {
						*wmaskp = true;
					// 横を比較(左,中央)
					} else if (is_threshold_over(wbluryp[        -1], wbluryp[0])) {
						*wmaskp = true;
					// 縦を比較(下,中央)
					} else if (is_threshold_over(wbluryp[+fpip->w  ], wbluryp[0])) {
						*wmaskp = true;
					// 斜め比較(右下,中央)
					} else if (is_threshold_over(wbluryp[+fpip->w+1], wbluryp[0])) {
						*wmaskp = true;
					// 横を比較(右,中央)
					} else if (is_threshold_over(wbluryp[        +1], wbluryp[0])) {
						*wmaskp = true;
					// 斜め比較(右上,中央)
					} else if (is_threshold_over(wbluryp[-fpip->w+1], wbluryp[0])) {
						*wmaskp = true;
					}
				}
			}
			wmaskp++;
			wbluryp++;
		}
	}

	//========================================
	// マスク結果の返却
	//========================================
	if (param.is_mask) {
		for (int y = y_start; y < y_end; y++ ) {
			// 処理対象のポインタの計算
			PIXEL_YC *ycp_temp = fpip->ycp_temp +  y            * max_w;	// テンポラリ領域
			bool     *wmaskp   = wmask          + (y - y_start) * wpitch;	//マスク

			// 横方向ループ
			for (int x = 0; x < fpip->w; x++ ) {
				// 色差はフィルタの対象外
				ycp_temp->cb = 0;
				ycp_temp->cr = 0;
				// 輝度に対する処理
				if (*wmaskp == true) {
					ycp_temp->y = 3072;
				} else {
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
	// 結果の適用
	//========================================
	for (int y = y_start; y < y_end; y++ ) {
		// 処理対象のポインタの計算
		PIXEL_YC *ycp      = fpip->ycp_edit + y * max_w;						// 画像データ
		PIXEL_YC *ycp_temp = fpip->ycp_temp + y * max_w;						// テンポラリ領域
		PIXEL_YC *wbluryp  = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;	//ブラー
		bool     *wmaskp   = wmask  + (y - y_start                 ) * wpitch;	//マスク

		// 横方向ループ
		for (int x = 0; x < fpip->w; x++ ) {
			// 色差はフィルタの対象外
			ycp_temp->cb = ycp->cb;
			ycp_temp->cr = ycp->cr;

			// 輝度に対する処理
			if (*wmaskp == true) {
				int y = max(min(ycp->y * 4 - wbluryp->y * 3, 4096), 0);
				ycp_temp->y = (param.strength * y + param.invstrength * ycp->y) >> 8;
			} else {
				ycp_temp->y = ycp->y;
			}

			//
			ycp++;
			ycp_temp++;
			wbluryp++;
			wmaskp++;
		}
	}

	//delete[] wblurx;
	//delete[] wblury;
	//delete[] wmask;
}

//---------------------------------------------------------------------
//		フィルタ処理関数
//---------------------------------------------------------------------
BOOL cpu_filter_func( 
	_In_ FILTER* fp,
	_In_ FILTER_PROC_INFO* fpip)
{
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
	fp->exfunc->exec_multi_thread_func(msharpen_get_thread_num,(void *)&thread_num, NULL);

	//========================================
	// ワークメモリの確保
	//========================================
	MSHARPEN_CORE_PARAM param;
//	param.thread_step = fpip->h / thread_num + BLURX_MARGIN_Y * 2;
	param.thread_step =
		(fpip->h / thread_num) + (fpip->h % thread_num != 0 ? 1 : 0) +
		BLURX_MARGIN_Y * 2;
	//	int wpitch = (fpip->w + 7) & (~7);
	param.wpitch = (fpip->w + 31) & (~31);			//2014.08.31 AVX2に向けて補正256bit=32byte
	param.wblurx = new PIXEL_YC[param.thread_step * thread_num * param.wpitch];
	param.wblury = new PIXEL_YC[param.thread_step * thread_num * param.wpitch];
	param.wmask  = new bool    [param.thread_step * thread_num * param.wpitch];

	//========================================
	// フィルターの実行
	//========================================
	fp->exfunc->exec_multi_thread_func(msharpen_core, (void *)&param, (void *)fpip);

	//========================================
	// ワークメモリの後始末
	//========================================
	delete[] param.wblurx;
	delete[] param.wblury;
	delete[] param.wmask;

	return TRUE;
}

#pragma endregion
