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
#include <filter_h_wrap.h>

//---------------------------------------------------------------------
//		関数プロトタイプ
//---------------------------------------------------------------------
extern BOOL cpu_filter_func(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip);
extern BOOL cpu_filter_avx2_func(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip);

//---------------------------------------------------------------------
//		フィルタ処理関数 スレッド数を返却
//---------------------------------------------------------------------
inline void msharpen_get_thread_num(
	_In_ int thread_id,
	_In_ int thread_num,
	_In_ void *param1,
	_In_ void *param2)
{
	*((int*)param1) = thread_num;
}

