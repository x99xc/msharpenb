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
//		�萔
//---------------------------------------------------------------------
// �`�F�b�N�{�b�N�X�̐ݒ�l
static const int CHECK_HIGH_QUALITY		= 0;	// �`�F�b�N - ���i��
static const int CHECK_MASK				= 1;	// �`�F�b�N - �}�X�N
static const int CHECK_GPU_USE			= 2;	// �`�F�b�N - GPU���g�p����
static const int CHECK_AVX2_USE			= 3;	// �`�F�b�N - AVX2���g�p����
static const int CHECK_SKIP_DROP_FRAME	= 4;	// �`�F�b�N - �h���b�v�t���[���`�F�b�N
// �g���b�N�o�[�̐ݒ�l
static const int TRACK_THRESHOLD		= 0;	// �g���b�N - 臒l;
static const int TRACK_STRENGTH			= 1;	// �g���b�N - ����;
// �u���[�}�[�W��
static const int BLUR_MARGIN			= 1;	// �ڂ��������͗אڂ̂�
static const int BLURX_MARGIN_X			= 1;	// �}�[�L���O�������ɏc�ɗ]����1�񂪕K�v�ɂȂ�
static const int BLURX_MARGIN_Y			= 2;	// �u���[�c�����ƃ}�[�L���O�������ɏc�ɗ]���ȍs��1���K�v�ɂȂ�
static const int BLURY_MARGIN_X			= 1;	// �}�[�L���O�������ɏc�ɗ]����1�񂪕K�v�ɂȂ�
static const int BLURY_MARGIN_Y			= 1;	// �}�[�L���O�������ɏc�ɗ]����1�s���K�v�ɂȂ�

//---------------------------------------------------------------------
//		�t�B���^�����\����
//---------------------------------------------------------------------
typedef struct tagMSHARPEN_PARAM
{
	// �t�B���^�p�����[�^
	int threshold;								// �t�B���^�\����->�g���b�N�o�[�̐ݒ�l[臒l]
	int strength;								// �t�B���^�\����->�g���b�N�o�[�̐ݒ�l[����]
	int invstrength;
	BOOL is_high_quality;						// �t�B���^�\����->�`�F�b�N�{�b�N�X�̐ݒ�l[���i��]
	BOOL is_mask;								// �t�B���^�\����->�`�F�b�N�{�b�N�X�̐ݒ�l[�}�X�N]
	BOOL is_avx2_use;							// �t�B���^�\����->�`�F�b�N�{�b�N�X�̐ݒ�l[AVX2���g�p����]

	// �\�[�X�摜�ݒ�
	int2 yc_dim;								// �\�[�X�摜�T�C�Y(x:fpip->w���Z�b�g,y:fpip->h���Z�b�g)
	int yc_ypitch;								// y�̍��W�ϊ��p(fpip->max_w���Z�b�g)

	// ���L�������̐ݒ�
	// - �\�[�X�L���b�V��
	int2 syc_dim;								// �摜�T�C�Y(x:��,y:�c) (z=0:�\�[�X�L���b�V��)
	int syc_zpitch;								// z�̍��W�ϊ��p
	// - �u���[��Ɨp
	int2 syc_blurx_dim;							// �u���[��Ɨp�@�摜�T�C�Y(x:��,y:�c)
	int syc_blurx_zpitch;						// z�̍��W�ϊ��p
	// - �u���[��Ɨp2
	int2 syc_blury_dim;							// �u���[��Ɨp�@�摜�T�C�Y(x:��,y:�c)
	int syc_blury_zpitch;						// z�̍��W�ϊ��p
	// - �}�X�N��Ɨp
	int2 sb_mask_dim;							// �}�X�N�p�@�摜�T�C�Y(x:��,y:�c)

} MSHARPEN_PARAM, *LPMSHARPEN_PARAM;
extern MSHARPEN_PARAM param;
