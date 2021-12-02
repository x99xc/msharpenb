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
#pragma once

#include "stdafx.h"
#include <filter_h_wrap.h>

//---------------------------------------------------------------------
//		定数
//---------------------------------------------------------------------
// チェックボックスの設定値
static const int CHECK_HIGH_QUALITY		= 0;	// チェック - 高品質
static const int CHECK_MASK				= 1;	// チェック - マスク
static const int CHECK_GPU_USE			= 2;	// チェック - GPUを使用する
static const int CHECK_AVX2_USE			= 3;	// チェック - AVX2を使用する
static const int CHECK_SKIP_DROP_FRAME	= 4;	// チェック - ドロップフレームチェック
// トラックバーの設定値
static const int TRACK_THRESHOLD		= 0;	// トラック - 閾値;
static const int TRACK_STRENGTH			= 1;	// トラック - 強さ;
// ブラーマージン
static const int BLUR_MARGIN			= 1;	// ぼかし処理は隣接のみ
static const int BLURX_MARGIN_X			= 1;	// マーキング処理時に縦に余分な1列が必要になる
static const int BLURX_MARGIN_Y			= 2;	// ブラー縦処理とマーキング処理時に縦に余分な行が1つずつ必要になる
static const int BLURY_MARGIN_X			= 1;	// マーキング処理時に縦に余分な1列が必要になる
static const int BLURY_MARGIN_Y			= 1;	// マーキング処理時に縦に余分な1行が必要になる

//---------------------------------------------------------------------
//		フィルタ条件構造体
//---------------------------------------------------------------------
typedef struct tagMSHARPEN_PARAM
{
	// フィルタパラメータ
	int threshold;								// フィルタ構造体->トラックバーの設定値[閾値]
	int strength;								// フィルタ構造体->トラックバーの設定値[強さ]
	int invstrength;
	BOOL is_high_quality;						// フィルタ構造体->チェックボックスの設定値[高品質]
	BOOL is_mask;								// フィルタ構造体->チェックボックスの設定値[マスク]
	BOOL is_avx2_use;							// フィルタ構造体->チェックボックスの設定値[AVX2を使用する]

	// ソース画像設定
	int2 yc_dim;								// ソース画像サイズ(x:fpip->wをセット,y:fpip->hをセット)
	int yc_ypitch;								// yの座標変換用(fpip->max_wをセット)

	// 共有メモリの設定
	// - ソースキャッシュ
	int2 syc_dim;								// 画像サイズ(x:横,y:縦) (z=0:ソースキャッシュ)
	int syc_zpitch;								// zの座標変換用
	// - ブラー作業用
	int2 syc_blurx_dim;							// ブラー作業用　画像サイズ(x:横,y:縦)
	int syc_blurx_zpitch;						// zの座標変換用
	// - ブラー作業用2
	int2 syc_blury_dim;							// ブラー作業用　画像サイズ(x:横,y:縦)
	int syc_blury_zpitch;						// zの座標変換用
	// - マスク作業用
	int2 sb_mask_dim;							// マスク用　画像サイズ(x:横,y:縦)

} MSHARPEN_PARAM, *LPMSHARPEN_PARAM;
extern MSHARPEN_PARAM param;
