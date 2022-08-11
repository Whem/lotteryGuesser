﻿// <auto-generated />
using LotteryMaui.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Migrations;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

#nullable disable

namespace LotteryMaui.Migrations
{
    [DbContext(typeof(DataContext))]
    [Migration("20220528094049_Init")]
    partial class Init
    {
        protected override void BuildTargetModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder.HasAnnotation("ProductVersion", "6.0.5");

            modelBuilder.Entity("LotteryMaui.Entities.LotteryNumber", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<int>("Message")
                        .HasColumnType("INTEGER");

                    b.Property<string>("Numbers")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<int>("WeekOfPull")
                        .HasColumnType("INTEGER");

                    b.HasKey("Id");

                    b.ToTable("LotteryNumbers");
                });

            modelBuilder.Entity("LotteryMaui.Entities.LotteryUser", b =>
                {
                    b.Property<int>("Id")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<string>("PasswordHash")
                        .HasColumnType("TEXT");

                    b.Property<string>("UserEmail")
                        .HasColumnType("TEXT");

                    b.Property<string>("UserName")
                        .HasColumnType("TEXT");

                    b.HasKey("Id");

                    b.ToTable("LotteryUsers");
                });
#pragma warning restore 612, 618
        }
    }
}
