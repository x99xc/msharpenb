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
/*
文章を書くのが苦手です。不明点があればツイッター@sundola8xにダイレクトメッセージで宜しくお願いします。
フィルタの処理内容に多少理解していない場合がございますので全てをお答えすることはできません。
このプログラムによる損失は当方では負いかねます。ご使用は自己責任にて行ってください。

MSharpenフィルタについて
    MSharpenaはAviutlのために再実装した、AviSynthの非常にシンプルなエッジ強調プラグインです。
    このフィルタは、古いneuron2のプラグインを移植したものです。
    このフィルタは判定用ブラー画像を生成してエッジ検出処理を行っているためノイズに強いと評判があります。

環境構築方法：
    Aviutl配下のフォルダに下記ファイルをコピーしてください。
        msvcr120.dll
        msvcp120.dll
        cudart32_65.dll
        cuda_manager.dll このDLLが新しくなっています。
    Aviutl配下のフォルダかpluginsフォルダに下記ファイルをコピーしてください。
        msharpenb.auf

動作環境(テスト環境)：
    OS     : Windows7 Ultimate 64bit SP1
    Aviutl : 0.99l
    CPU    : Intel Core i7-4790K 4.0GHz
    Mem    : 16GB
    GPU    : NVIDIA Geforce GTX 780Ti
             NVIDIA製GPUドライバー、340.62 (古いバージョンでも動作するかもしれません。)
             GPUはNVIDIA Geforce GTX 560Ti 以上推奨　それ以外でも動く可能性あり。テスター募集

設定値の説明：(本家説明を引用)
    閾値 : (0-255、デフォルト:15)
        強調するエッジを見つけるために使用されれます。値を小さくすると強調する箇所が増えます。
        強調される個所を確認するには「マスクモード」にチェックを入れてください。
    強さ : (0-255、デフォルト:100)
        強調するエッジに適用される先鋭化の強さ。値を大きくするとエッジの強調度が増えます。
    高品質モード : (デフォルト:オフ)
        チェック入れるとエッジを検出判定処理を増やして精度を高めます。若干重くなります。
    マスクモード : (デフォルト:オフ)
        チェックを入れると強調される個所を確認することができます。
    GPU(CUDA)を使用する : (デフォルト:オフ)
        チェックを入れるとフィルタ処理をGPUにやらせます。チェックを外すとフィルタ処理をCPUにやらせます。
　　　　NVIDIA GTX 780Ti 以上のGPUでパフォーマンスが改善しています。
    AVX2を使用する : (デフォルト:オフ)
　　　　チェックを入れるとフィルタ処理にAVX2を使用します。
　　　　注意！！！AVX2が使用できるかチェックをしていませんので自己責任でチェックを入れてください。
　　　　　　　　　「 GPU(CUDA)を使用する」にチェックが入っている場合はそっちが優先されます。
　　　　AVX2を使用した場合の処理結果は、他のフィルタと若干誤差が生じますのでご了承ください。
    ドロップフレームチェック(テスト) : (デフォルト:オフ)
        チェックを入れると自動フィールドシフト動作時のドロップフレームで、
        フィルタの処理をスキップするため無駄な処理を減らせます。
        ただし、テスト実相なので自動フィールドシフトの実ドロップフレームと
        判定が異なることがあります。オフ推奨です。

本家サイトURL https://github.com/tp7/msharpen

更新履歴
  Ver.0.0.1 -- test version
  Ver.0.0.2 -- 地味に高速化
*/
#include "stdafx.h"
#include "msharpenb.h"
#include <msharpenbLibCUDA20_cuda.h>
#include <msharpenbLibCUDA35_cuda.h>
#include "msharpenb_cpu.h"
#include <afs_check.h>

//---------------------------------------------------------------------
//		関数プロトタイプ
//---------------------------------------------------------------------
extern BOOL func_proc(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip);
extern BOOL func_init(_In_ FILTER* fp);
extern BOOL func_exit(_In_ FILTER* fp);
extern BOOL func_update(_In_ FILTER *fp, _In_ int status );
//---------------------------------------------------------------------
//		フィルタ構造体定義
//---------------------------------------------------------------------
// タイトル
static const LPSTR lpszFilterTilte = "msharpen +beta";
static const LPSTR lpszFilterCOMM = "msharpen +beta Ver.0.0.2 MT @sundola8x";
// 定義
static int TRACK_N = 2;														//	トラックバーの数
static TCHAR* track_name[] ={"閾値","強さ"	};								//	トラックバーの名前
static int track_default[] ={15,	100,	};								//	トラックバーの初期値
static int track_s[] =		{0,		0,		};								//	トラックバーの下限値
static int track_e[] =		{255,	255,	};								//	トラックバーの上限値
static int CHECK_N = 5;														//	チェックボックスの数
static TCHAR* check_name[] ={												//	チェックボックスの名前
	"高品質モード",
	"マスクモード",
	"GPU(CUDA)を使用する",
	"AVX2を使用する",
	"ドロップフレームチェック(テスト)" };
static int check_default[] ={												//	チェックボックスの初期値 (値は0か1)
	FALSE,		//	高品質モード
	FALSE,		//	マスクモード
	FALSE,		//	GPU(CUDA)を使用する
	FALSE,		//	AVX2を使用する
	FALSE };	//	ドロップフレームチェック(テスト)

static FILTER_DLL filter = {
	FILTER_FLAG_EX_INFORMATION,
	0,0,
	lpszFilterTilte,
	TRACK_N,
	track_name,
	track_default,
	track_s,track_e,
	CHECK_N,
	check_name,
	check_default,
	func_proc,
	func_init,//
	func_exit,//
	func_update,//
	NULL,//
	NULL,NULL,
	NULL,
	NULL,
	lpszFilterCOMM,
	NULL,
	NULL,
};

//---------------------------------------------------------------------
//		グローバル
//---------------------------------------------------------------------
// どこかに保存できればいいのだが
MSHARPEN_PARAM param;
// 使用するCPUフィルタ関数を保存
BOOL(*use_cpu_filter_func)(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip);
// 使用するGPUフィルタ関数を保存
BOOL(*use_cuda_filter_func)(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip, _Inout_ MSHARPEN_PARAM* paramp);

#pragma region **** フィルタ初期化 ****

//---------------------------------------------------------------------
// フィルタ基礎データの構築
//---------------------------------------------------------------------
inline BOOL __stdcall _init_param(
	_In_ const FILTER* fp)
{
	// ========================================
	// フィルタの設定を格納
	// ========================================
	param.threshold = fp->track[TRACK_THRESHOLD] << 4;	// フィルタ構造体->トラックバーの設定値[閾値]
	param.strength = fp->track[TRACK_STRENGTH];		// フィルタ構造体->トラックバーの設定値[強さ]
	param.invstrength = 255 - fp->track[TRACK_STRENGTH];	// フィルタ構造体->トラックバーの設定値[強さ]
	param.is_high_quality = fp->check[CHECK_HIGH_QUALITY];	// フィルタ構造体->チェックボックスの設定値[高品質]
	param.is_mask = fp->check[CHECK_MASK];			// フィルタ構造体->チェックボックスの設定値[マスク]

	// ========================================
	// 使用するCPUフィルタ関数を保存
	// ========================================
	if ((param.is_avx2_use == TRUE && fp->check[CHECK_AVX2_USE] == FALSE) || 
		(param.is_avx2_use == FALSE && fp->check[CHECK_AVX2_USE] == TRUE) ||
		use_cpu_filter_func == NULL) {
		if (fp->check[CHECK_AVX2_USE] == TRUE) {
			if (get_availableSIMD() & AVX2 != 0) {
				DBG("using cpu_filter_avx2_func \n");
				use_cpu_filter_func = cpu_filter_avx2_func;
				param.is_avx2_use = TRUE;
			}
			else {
				DBG("using cpu_filter_func \n");
				use_cpu_filter_func = cpu_filter_func;
				param.is_avx2_use = FALSE;
				fp->check[CHECK_AVX2_USE] = FALSE;
				fp->exfunc->filter_window_update((void*)fp);
			}
		}
		else {
			DBG("using cpu_filter_func \n");
			use_cpu_filter_func = cpu_filter_func;
			param.is_avx2_use = FALSE;
		}
	}

	return  TRUE;
}

#pragma endregion

#pragma region **** 外部呼出し用 ****

//---------------------------------------------------------------------
//		フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable( void )
{
	return &filter;
}

//---------------------------------------------------------------------
//		フィルタ初期化関数
//---------------------------------------------------------------------
BOOL func_init( 
	_In_ FILTER* fp)
{
	//========================================
	// チェック
	//========================================
	assert(fp != NULL);
	assert(fp->track != NULL);
	assert(fp->check != NULL);

	//========================================
	// デバッグ 初期化
	//========================================
	DBG_INIT();

	// ========================================
	// フィルタ基礎データの構築
	// ========================================
	ZeroMemory(&param, sizeof(param));
	_init_param(fp);

	return TRUE;
}

//---------------------------------------------------------------------
//		フィルタ後始末関数
//---------------------------------------------------------------------
BOOL func_exit( 
	_In_ FILTER* fp)
{
	//========================================
	// チェック
	//========================================
	assert(fp != NULL);
	assert(fp->track != NULL);
	assert(fp->check != NULL);

	//========================================
	// CUDA 後始末
	//========================================
	//cuda_manager_exit();

	//========================================
	// デバッグ用 後始末
	//========================================
	DBG_EXIT();

	return TRUE;
}

//---------------------------------------------------------------------
//		フィルタ更新関数
//---------------------------------------------------------------------
BOOL func_update(
	_In_ FILTER *fp,
	_In_ int status)
{
	// ========================================
	// フィルタの設定が変わるたびに呼び出されるため
	// 設定に変更がない場合に変化がないデータをここで更新すると効率的
	// ========================================
	DBG("update %8x\n", status);

	// ========================================
	// フィルタ基礎データの構築
	// ========================================
	// 全ての設定が変更された場合
	if (status == FILTER_UPDATE_STATUS_ALL) {
		DBG("_init_param \n");
		_init_param(fp);
	// トラックバーの設定が変更された場合
	} else if (status & FILTER_UPDATE_STATUS_TRACK) {
		DBG("_init_param \n");
		_init_param(fp);
	// チェックボックスの設定が変更された場合
	} else if (status & FILTER_UPDATE_STATUS_CHECK) {
		DBG("_init_param \n");
		_init_param(fp);
	}

	return TRUE;
}

//---------------------------------------------------------------------
//		フィルタ処理関数
//---------------------------------------------------------------------
BOOL func_proc( 
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

	// ========================================
	// ドロップフレームチェック(テスト)
	// ========================================
	if (fp->check[CHECK_SKIP_DROP_FRAME] && is_afs_drop_frame(fp, fpip)) {
		return TRUE;
	}

	//========================================
	// フィルターの切り替え
	//========================================
	if (fp->check[CHECK_GPU_USE] == TRUE && 
		cuda_manager_init(fp, fpip) == TRUE) { 
        // 使用フィルタを選定
		if (use_cuda_filter_func == NULL) {
			if (get_cuda_compute_capability_major() > 3 || 
				(get_cuda_compute_capability_major() == 3 && get_cuda_compute_capability_minor() >= 5)) {
				DBG("using cuda_filter_CUDA35_func \n");
				use_cuda_filter_func = cuda_filter_CUDA35_func;
			}
			else {
				DBG("using cuda_filter_CUDA20_func \n");
				use_cuda_filter_func = cuda_filter_CUDA20_func;
			}
		}

		if (use_cuda_filter_func(fp, fpip, &param) == FALSE)
			return FALSE;
	} else {
		if (use_cpu_filter_func(fp, fpip) == FALSE)
			return FALSE;
	}

	//========================================
	// アドレス交換
	//========================================
	PIXEL_YC* temp = fpip->ycp_edit;
	fpip->ycp_edit = fpip->ycp_temp;
	fpip->ycp_temp = temp;

	return TRUE;
}

#pragma endregion
